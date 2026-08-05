//! The server-side API surface: construction, discipline swaps, the shared
//! `DisciplineService` RPCs, and the implicit server's discrete paths.
//!
//! Most cases call the service methods in process rather than over a socket —
//! everything except the streaming RPCs is reachable that way, and it keeps the
//! assertion on the server's own behaviour rather than on the client's decoding.

mod common;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use philote_mdo::client::ExplicitClient;
use philote_mdo::discrete::{json_to_value, value_to_json};
use philote_mdo::examples::{scalar, Paraboloid, QuadraticImplicit};
use philote_mdo::philote_info::{
    discipline_service_server::{DisciplineService, DisciplineServiceServer},
    explicit_service_server::ExplicitServiceServer,
    implicit_service_server::ImplicitServiceServer,
    DataType, StreamOptions as ProtoStreamOptions, VariableMetaData, VariableType,
};
use philote_mdo::registry::VariableRegistry;
use philote_mdo::server::{DisciplineServer, ExplicitServer, ImplicitServer};
use philote_mdo::traits::{Discipline, ExplicitDiscipline, ImplicitDiscipline};
use philote_mdo::{impl_registry, ArrayMap, DiscreteMap, PartialMap, PhiloteError, Result};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tonic::{Code, Request};

// --- Test disciplines ---

/// Scales `x` by a gain that arrives through `SetOptions`; declares its options in
/// `initialize`, which is what `DisciplineServer::new` is expected to run.
struct Scaler {
    registry: VariableRegistry,
    gain: f64,
}

impl Scaler {
    fn new(gain: f64) -> Self {
        Self {
            registry: VariableRegistry::default(),
            gain,
        }
    }
}

impl Discipline for Scaler {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "Scaler"
    }

    fn initialize(&mut self) -> Result<()> {
        self.add_option("gain", "float")?;
        self.add_option("label", "str")
    }

    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(gain) = options.get("gain").and_then(|v| v.as_f64()) {
            self.gain = gain;
        }
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("y", "x")
    }
}

#[async_trait]
impl ExplicitDiscipline for Scaler {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), &inputs["x"] * self.gain);
        Ok(outputs)
    }
}

/// Reports option types using the aliases the server accepts alongside the
/// canonical names. `add_option` refuses the aliases, so they can only reach the
/// server by overriding `get_available_options` — which is exactly what a
/// discipline ported from another Philote implementation would do.
#[derive(Default)]
struct AliasOptions {
    registry: VariableRegistry,
}

impl Discipline for AliasOptions {
    impl_registry!(registry);

    fn setup(&mut self) -> Result<()> {
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }

    fn get_available_options(&self) -> Result<HashMap<String, String>> {
        Ok([
            ("a_bool", "bool"),
            ("an_int", "int"),
            ("a_float", "float"),
            ("a_double", "double"),
            ("a_str", "str"),
            ("a_string", "string"),
            ("a_dict", "dict"),
            ("a_struct", "struct"),
        ]
        .iter()
        .map(|(name, ty)| (name.to_string(), ty.to_string()))
        .collect())
    }
}

#[async_trait]
impl ExplicitDiscipline for AliasOptions {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), inputs["x"].clone());
        Ok(outputs)
    }
}

/// Which lifecycle hook a [`Failing`] discipline should fail in.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum FailIn {
    #[default]
    Nothing,
    Options,
    UnknownOptionType,
    Configure,
    Setup,
    SetupPartials,
    VariableDefinitions,
    DiscreteDefinitions,
    PartialsDefinitions,
}

/// Fails in exactly one lifecycle hook, with a message naming that hook, so a test
/// can tell which error the RPC actually surfaced.
#[derive(Default)]
struct Failing {
    registry: VariableRegistry,
    fail: FailIn,
}

impl Failing {
    fn new(fail: FailIn) -> Self {
        Self {
            registry: VariableRegistry::default(),
            fail,
        }
    }

    fn boom(hook: &str) -> PhiloteError {
        PhiloteError::array_error(format!("{hook} deliberately failed"))
    }
}

impl Discipline for Failing {
    impl_registry!(registry);

    fn configure(&mut self) -> Result<()> {
        if self.fail == FailIn::Configure {
            return Err(Self::boom("configure"));
        }
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        if self.fail == FailIn::Setup {
            return Err(Self::boom("setup"));
        }
        self.add_input("x", &[1], "")?;
        self.add_output("y", &[1], "")
    }

    fn setup_partials(&mut self) -> Result<()> {
        if self.fail == FailIn::SetupPartials {
            return Err(Self::boom("setup_partials"));
        }
        self.declare_partials("y", "x")
    }

    fn get_available_options(&self) -> Result<HashMap<String, String>> {
        match self.fail {
            FailIn::Options => Err(Self::boom("get_available_options")),
            FailIn::UnknownOptionType => Ok(HashMap::from([(
                "weird".to_string(),
                "complex".to_string(),
            )])),
            _ => Ok(HashMap::new()),
        }
    }

    fn get_variable_definitions(&self) -> Result<Vec<VariableMetaData>> {
        if self.fail == FailIn::VariableDefinitions {
            return Err(Self::boom("get_variable_definitions"));
        }
        Ok(self.registry.var_meta().to_vec())
    }

    fn get_discrete_variable_definitions(&self) -> Result<Vec<VariableMetaData>> {
        if self.fail == FailIn::DiscreteDefinitions {
            return Err(Self::boom("get_discrete_variable_definitions"));
        }
        Ok(self.registry.discrete_meta().to_vec())
    }

    fn get_partials_definitions(&self) -> Result<Vec<(String, String)>> {
        if self.fail == FailIn::PartialsDefinitions {
            return Err(Self::boom("get_partials_definitions"));
        }
        Ok(self.registry.partials_meta().to_vec())
    }
}

#[async_trait]
impl ExplicitDiscipline for Failing {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let mut outputs = HashMap::new();
        outputs.insert("y".to_string(), inputs["x"].clone());
        Ok(outputs)
    }
}

/// `R(x) = x - (factor * a + offset)`, where `factor` arrives as a discrete input.
///
/// Every continuous entry point is `unreachable!`, so a test failing to take the
/// discrete branch shows up as a panic rather than a plausible-looking number.
#[derive(Default)]
struct ScaledImplicit {
    registry: VariableRegistry,
    offset: f64,
}

impl ScaledImplicit {
    fn with_offset(offset: f64) -> Self {
        Self {
            registry: VariableRegistry::default(),
            offset,
        }
    }

    fn factor(discrete_inputs: &DiscreteMap) -> f64 {
        discrete_inputs
            .get("factor")
            .map(|v| value_to_json(v).as_f64().unwrap_or(1.0))
            .unwrap_or(1.0)
    }

    fn note(factor: f64) -> DiscreteMap {
        HashMap::from([(
            "note".to_string(),
            json_to_value(&serde_json::json!(format!("factor={factor}"))),
        )])
    }
}

impl Discipline for ScaledImplicit {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "ScaledImplicit"
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("a", &[1], "")?;
        // The residual twin only appears because ImplicitServer marks the registry.
        self.add_output("x", &[1], "")?;
        self.add_discrete_input("factor", Some(json_to_value(&serde_json::json!(2))))?;
        self.add_discrete_output("note", None)
    }

    fn setup_partials(&mut self) -> Result<()> {
        self.declare_partials("x", "a")?;
        self.declare_partials("x", "x")
    }
}

#[async_trait]
impl ImplicitDiscipline for ScaledImplicit {
    async fn compute_residuals(&self, _inputs: &ArrayMap, _outputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn solve_residuals(&self, _inputs: &ArrayMap) -> Result<ArrayMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn residual_partials(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
    ) -> Result<PartialMap> {
        unreachable!("the server uses the discrete path when discrete variables exist")
    }

    async fn compute_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let factor = Self::factor(discrete_inputs);
        let residual = &outputs["x"] - &(&inputs["a"] * factor + self.offset);
        Ok((
            HashMap::from([("x".to_string(), residual)]),
            Self::note(factor),
        ))
    }

    async fn solve_residuals_with_discrete(
        &self,
        inputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<(ArrayMap, DiscreteMap)> {
        let factor = Self::factor(discrete_inputs);
        Ok((
            HashMap::from([("x".to_string(), &inputs["a"] * factor + self.offset)]),
            Self::note(factor),
        ))
    }

    async fn residual_partials_with_discrete(
        &self,
        _inputs: &ArrayMap,
        _outputs: &ArrayMap,
        discrete_inputs: &DiscreteMap,
    ) -> Result<PartialMap> {
        let factor = Self::factor(discrete_inputs);
        Ok(HashMap::from([
            (("x".to_string(), "a".to_string()), scalar(-factor)),
            (("x".to_string(), "x".to_string()), scalar(1.0)),
        ]))
    }
}

// --- Helpers ---

/// A running server whose server object stays reachable, so a test can swap the
/// served discipline while clients are connected. The shared harness in
/// `tests/common` hands the server to tonic, which is enough for every other test.
struct SharedServer<S> {
    addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    handle: Option<JoinHandle<()>>,
    server: Arc<S>,
}

impl<S> SharedServer<S> {
    fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }

    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl<S> Drop for SharedServer<S> {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

async fn bind() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind an ephemeral loopback port");
    let addr = listener.local_addr().expect("resolve the bound address");
    (listener, addr)
}

async fn spawn_shared_explicit<D: ExplicitDiscipline + 'static>(
    discipline: D,
) -> SharedServer<ExplicitServer<D>> {
    let (listener, addr) = bind().await;
    let (tx, rx) = oneshot::channel();
    let server = Arc::new(ExplicitServer::new(discipline));

    let serving = Arc::clone(&server);
    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(DisciplineServiceServer::from_arc(serving.clone()))
            .add_service(ExplicitServiceServer::from_arc(serving))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = rx.await;
            })
            .await;
    });

    SharedServer {
        addr,
        shutdown: Some(tx),
        handle: Some(handle),
        server,
    }
}

async fn spawn_shared_implicit<D: ImplicitDiscipline + 'static>(
    discipline: D,
) -> SharedServer<ImplicitServer<D>> {
    let (listener, addr) = bind().await;
    let (tx, rx) = oneshot::channel();
    let server = Arc::new(ImplicitServer::new(discipline));

    let serving = Arc::clone(&server);
    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(DisciplineServiceServer::from_arc(serving.clone()))
            .add_service(ImplicitServiceServer::from_arc(serving))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = rx.await;
            })
            .await;
    });

    SharedServer {
        addr,
        shutdown: Some(tx),
        handle: Some(handle),
        server,
    }
}

/// The options a server advertises, as a name-to-`DataType` map.
async fn advertised_options<S: DisciplineService>(server: &S) -> HashMap<String, DataType> {
    let list = server
        .get_available_options(Request::new(()))
        .await
        .expect("GetAvailableOptions should succeed")
        .into_inner();
    list.options
        .into_iter()
        .zip(list.r#type)
        .map(|(name, code)| {
            (
                name,
                DataType::try_from(code).expect("a valid wire data type"),
            )
        })
        .collect()
}

fn status_of(err: PhiloteError) -> tonic::Status {
    match err {
        PhiloteError::GrpcError(status) => *status,
        other => panic!("expected a gRPC error, got {other:?}"),
    }
}

fn inputs_x(value: f64) -> ArrayMap {
    HashMap::from([("x".to_string(), scalar(value))])
}

// --- Construction and discipline replacement ---

#[tokio::test]
async fn constructing_a_server_runs_initialize_so_declared_options_are_advertised() {
    // A discipline built with Default has an empty options list until initialize
    // runs; the server runs it so clients can discover options before Setup.
    let server = ExplicitServer::new(Scaler::new(2.0));

    let options = advertised_options(&server).await;
    assert_eq!(options.len(), 2);
    assert_eq!(options["gain"], DataType::KDouble);
    assert_eq!(options["label"], DataType::KString);
}

#[tokio::test]
async fn replacing_the_discipline_does_not_re_run_initialize() {
    // Documented contract: set_discipline installs the discipline as given. A
    // freshly constructed Scaler has not had initialize run, so nothing is
    // advertised until the caller runs it.
    let server = ExplicitServer::new(Scaler::new(2.0));
    assert_eq!(advertised_options(&server).await.len(), 2);

    server.set_discipline(Scaler::new(3.0)).await;
    assert!(
        advertised_options(&server).await.is_empty(),
        "set_discipline must not call initialize on the replacement"
    );
}

#[tokio::test]
async fn replacing_the_discipline_changes_what_later_computes_return() {
    let server = spawn_shared_explicit(Scaler::new(2.0)).await;
    let mut client = ExplicitClient::connect(server.endpoint())
        .await
        .expect("connect to the test server");
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    assert_eq!(
        client.compute_function(&inputs_x(3.0)).await.unwrap()["y"][[0]],
        6.0
    );

    server.server.set_discipline(Scaler::new(10.0)).await;

    // The replacement arrives with an empty registry, so the handshake is redone.
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    assert_eq!(
        client.compute_function(&inputs_x(3.0)).await.unwrap()["y"][[0]],
        30.0,
        "the swapped-in discipline should serve subsequent computes"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn replacing_an_implicit_discipline_changes_what_later_solves_return() {
    let server = spawn_shared_implicit(ScaledImplicit::default()).await;
    let mut client = philote_mdo::client::ImplicitClient::connect(server.endpoint())
        .await
        .expect("connect to the test server");
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    let inputs = HashMap::from([("a".to_string(), scalar(4.0))]);
    // Default factor of 2, no offset.
    assert_eq!(
        client.solve_residuals(&inputs).await.unwrap()["x"][[0]],
        8.0
    );

    server
        .server
        .set_discipline(ScaledImplicit::with_offset(100.0))
        .await;

    // The replacement arrives with an empty registry, so the handshake is redone.
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();
    client.get_partial_definitions().await.unwrap();

    assert_eq!(
        client.solve_residuals(&inputs).await.unwrap()["x"][[0]],
        108.0,
        "the swapped-in discipline should serve subsequent solves"
    );

    server.shutdown().await;
}

#[tokio::test]
async fn the_implicit_server_marks_the_registry_so_outputs_get_residual_twins() {
    // ScaledImplicit builds a plain (explicit-style) registry; without the server's
    // mark_implicit its output would arrive with no residual to report.
    let server = ImplicitServer::new(ScaledImplicit::default());
    assert!(server.discipline().read().await.registry().is_implicit());

    server.setup(Request::new(())).await.expect("Setup");

    let discipline = server.discipline().read().await;
    let residuals: Vec<_> = discipline
        .registry()
        .var_meta()
        .iter()
        .filter(|v| v.r#type == i32::from(VariableType::KResidual))
        .collect();
    assert_eq!(residuals.len(), 1);
    assert_eq!(residuals[0].name, "x");
}

#[tokio::test]
async fn verbose_logging_does_not_change_what_a_server_reports() {
    // with_verbose only adds tracing output; every observable answer must match.
    let quiet = ExplicitServer::new(Scaler::new(2.0));
    let loud = ExplicitServer::new(Scaler::new(2.0)).with_verbose(true);
    assert_eq!(
        DisciplineService::get_info(&quiet, Request::new(()))
            .await
            .unwrap()
            .into_inner(),
        DisciplineService::get_info(&loud, Request::new(()))
            .await
            .unwrap()
            .into_inner()
    );
    assert_eq!(
        advertised_options(&quiet).await,
        advertised_options(&loud).await
    );

    let quiet = ImplicitServer::new(QuadraticImplicit::new());
    let loud = ImplicitServer::new(QuadraticImplicit::new()).with_verbose(true);
    assert_eq!(
        DisciplineService::get_info(&quiet, Request::new(()))
            .await
            .unwrap()
            .into_inner(),
        DisciplineService::get_info(&loud, Request::new(()))
            .await
            .unwrap()
            .into_inner()
    );
}

#[test]
fn the_base_server_reports_its_verbose_setting() {
    assert!(!DisciplineServer::new(Paraboloid::new()).verbose());
    assert!(DisciplineServer::new(Paraboloid::new())
        .with_verbose(true)
        .verbose());
}

// --- Stream options ---

#[tokio::test]
async fn a_non_positive_chunk_size_is_rejected_and_leaves_the_setting_untouched() {
    let base = DisciplineServer::new(Paraboloid::new());
    let default_size = base.stream_options().read().await.max_double_per_slice;

    for bad in [0, -8] {
        let status = base
            .set_stream_options(Request::new(ProtoStreamOptions { num_double: bad }))
            .await
            .expect_err("a non-positive num_double should be rejected");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(
            status.message().contains("num_double"),
            "the message should name the offending field: {}",
            status.message()
        );
        assert_eq!(
            base.stream_options().read().await.max_double_per_slice,
            default_size,
            "a rejected request must not clobber the negotiated chunk size"
        );
    }
}

#[tokio::test]
async fn an_accepted_chunk_size_becomes_the_setting_in_effect() {
    let base = DisciplineServer::new(Paraboloid::new());
    base.set_stream_options(Request::new(ProtoStreamOptions { num_double: 7 }))
        .await
        .expect("a positive num_double should be accepted");
    assert_eq!(base.stream_options().read().await.max_double_per_slice, 7);
}

#[tokio::test]
async fn a_small_chunk_size_still_returns_the_whole_array() {
    // The negotiated size governs how the server splits its response, so a size
    // below the array length must still reassemble into the same gradient.
    let server = common::spawn_explicit(philote_mdo::examples::Rosenbrock::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let inputs = HashMap::from([("x".to_string(), philote_mdo::examples::vector(&[1.5, 2.0]))]);
    let key = ("f".to_string(), "x".to_string());

    let whole = client.compute_gradient(&inputs).await.unwrap();
    assert_eq!(whole[&key].len(), 2);
    assert!(
        whole[&key].iter().any(|v| v.abs() > 1e-6),
        "the gradient away from the optimum should be non-trivial"
    );

    client
        .set_stream_options(philote_mdo::types::StreamOptions {
            max_double_per_slice: 1,
        })
        .await
        .unwrap();
    let chunked = client.compute_gradient(&inputs).await.unwrap();
    assert_eq!(
        chunked[&key], whole[&key],
        "one value per message must reassemble into the same gradient"
    );

    server.shutdown().await;
}

// --- Option type mapping ---

#[tokio::test]
async fn option_type_aliases_map_to_the_canonical_wire_types() {
    let server = ExplicitServer::new(AliasOptions::default());
    let options = advertised_options(&server).await;

    assert_eq!(options["a_bool"], DataType::KBool);
    assert_eq!(options["an_int"], DataType::KInt);
    assert_eq!(options["a_float"], DataType::KDouble);
    assert_eq!(options["a_double"], DataType::KDouble);
    assert_eq!(options["a_str"], DataType::KString);
    assert_eq!(options["a_string"], DataType::KString);
    assert_eq!(options["a_dict"], DataType::KStruct);
    assert_eq!(options["a_struct"], DataType::KStruct);
}

#[tokio::test]
async fn option_type_aliases_reach_the_client_under_their_canonical_names() {
    let server = common::spawn_explicit(AliasOptions::default()).await;
    let mut client = ExplicitClient::connect(server.endpoint()).await.unwrap();

    let options = client.get_available_options().await.unwrap();
    assert_eq!(options["a_double"], "float");
    assert_eq!(options["a_string"], "str");
    assert_eq!(options["a_struct"], "dict");
    // The canonical spellings survive the same round trip.
    assert_eq!(options["a_float"], "float");
    assert_eq!(options["a_str"], "str");
    assert_eq!(options["a_dict"], "dict");

    server.shutdown().await;
}

#[tokio::test]
async fn an_option_with_an_unrecognised_type_is_rejected() {
    let server = ExplicitServer::new(Failing::new(FailIn::UnknownOptionType));
    let status = server
        .get_available_options(Request::new(()))
        .await
        .expect_err("an unknown option type should be rejected");

    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("weird") && status.message().contains("complex"),
        "the message should name the option and its bad type: {}",
        status.message()
    );
}

// --- Lifecycle error paths ---

#[tokio::test]
async fn a_failing_options_lookup_surfaces_from_get_available_options() {
    let server = ExplicitServer::new(Failing::new(FailIn::Options));
    let status = server
        .get_available_options(Request::new(()))
        .await
        .expect_err("the discipline error should surface");
    assert!(status
        .message()
        .contains("get_available_options deliberately failed"));
}

#[tokio::test]
async fn setup_reports_which_lifecycle_hook_failed() {
    for (fail, hook) in [
        (FailIn::Configure, "configure"),
        (FailIn::Setup, "setup"),
        (FailIn::SetupPartials, "setup_partials"),
    ] {
        let server = ExplicitServer::new(Failing::new(fail));
        let status = DisciplineService::setup(&server, Request::new(()))
            .await
            .expect_err("a failing hook should fail the Setup RPC");
        assert!(
            status
                .message()
                .contains(&format!("{hook} deliberately failed")),
            "expected the {hook} failure, got: {}",
            status.message()
        );
    }
}

#[tokio::test]
async fn get_variable_definitions_surfaces_continuous_and_discrete_failures() {
    for (fail, hook) in [
        (FailIn::VariableDefinitions, "get_variable_definitions"),
        (
            FailIn::DiscreteDefinitions,
            "get_discrete_variable_definitions",
        ),
    ] {
        let server = ExplicitServer::new(Failing::new(fail));
        let status = DisciplineService::get_variable_definitions(&server, Request::new(()))
            .await
            .err()
            .expect("a failing definition lookup should fail the RPC");
        assert!(
            status
                .message()
                .contains(&format!("{hook} deliberately failed")),
            "expected the {hook} failure, got: {}",
            status.message()
        );
    }
}

#[tokio::test]
async fn get_partial_definitions_surfaces_a_discipline_failure() {
    let server = ExplicitServer::new(Failing::new(FailIn::PartialsDefinitions));
    let status = DisciplineService::get_partial_definitions(&server, Request::new(()))
        .await
        .err()
        .expect("a failing partials lookup should fail the RPC");
    assert!(status
        .message()
        .contains("get_partials_definitions deliberately failed"));
}

#[tokio::test]
async fn set_variable_shapes_rejects_an_unknown_variable_type() {
    let server = common::spawn_explicit(Paraboloid::new()).await;
    let mut client = common::connected_explicit_client(&server).await;

    let err = client
        .send_variable_shapes(vec![VariableMetaData {
            r#type: 999,
            name: "x".to_string(),
            shape: vec![2],
            units: String::new(),
            dynamic_shape: true,
        }])
        .await
        .expect_err("an out-of-range variable type should be rejected");

    let status = status_of(err);
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(
        status.message().contains("999") && status.message().contains('x'),
        "the message should name the bad type and variable: {}",
        status.message()
    );

    server.shutdown().await;
}

// --- Preallocation ---

#[tokio::test]
async fn partial_preallocation_uses_the_declared_variable_shapes() {
    let base = DisciplineServer::new(Paraboloid::new());
    base.setup(Request::new(())).await.expect("Setup");

    let partials = base
        .preallocate_partials()
        .await
        .expect("declared partials should preallocate");

    assert_eq!(partials.len(), 2);
    for var in ["x", "y"] {
        let array = &partials[&("f_xy".to_string(), var.to_string())];
        assert_eq!(array.shape(), &[1]);
        assert_eq!(array[[0]], 0.0, "preallocated partials start zeroed");
    }
}

#[tokio::test]
async fn preallocation_fails_before_setup_declares_anything() {
    // No Setup means no shapes, so a compute would scatter into nothing; the
    // failure has to come from preallocation rather than surface as empty results.
    let base = DisciplineServer::new(Failing::new(FailIn::PartialsDefinitions));
    assert!(base.preallocate_inputs().await.unwrap().is_empty());
    assert!(base.preallocate_outputs().await.unwrap().is_empty());
}

// --- Implicit discrete paths ---

#[tokio::test]
async fn discrete_inputs_reach_the_implicit_residual_evaluation() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let inputs = HashMap::from([("a".to_string(), scalar(3.0))]);
    let discrete = HashMap::from([("factor".to_string(), json_to_value(&serde_json::json!(5)))]);

    // At x = 15 the residual is zero for factor 5, and non-zero away from it.
    let (residuals, discrete_outputs) = client
        .compute_residuals_with_discrete(
            &inputs,
            &HashMap::from([("x".to_string(), scalar(15.0))]),
            &discrete,
        )
        .await
        .unwrap();
    assert_eq!(residuals["x"][[0]], 0.0);
    assert_eq!(
        value_to_json(&discrete_outputs["note"]),
        serde_json::json!("factor=5")
    );

    let (residuals, _) = client
        .compute_residuals_with_discrete(
            &inputs,
            &HashMap::from([("x".to_string(), scalar(10.0))]),
            &discrete,
        )
        .await
        .unwrap();
    assert_eq!(residuals["x"][[0]], -5.0);

    server.shutdown().await;
}

#[tokio::test]
async fn discrete_inputs_reach_the_implicit_solve() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let inputs = HashMap::from([("a".to_string(), scalar(3.0))]);
    let discrete = HashMap::from([("factor".to_string(), json_to_value(&serde_json::json!(5)))]);

    let (outputs, discrete_outputs) = client
        .solve_residuals_with_discrete(&inputs, &discrete)
        .await
        .unwrap();

    assert_eq!(outputs["x"][[0]], 15.0);
    assert_eq!(
        value_to_json(&discrete_outputs["note"]),
        serde_json::json!("factor=5")
    );

    server.shutdown().await;
}

#[tokio::test]
async fn discrete_inputs_reach_the_implicit_residual_gradients() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    let partials = client
        .compute_residual_gradients_with_discrete(
            &HashMap::from([("a".to_string(), scalar(3.0))]),
            &HashMap::from([("x".to_string(), scalar(15.0))]),
            &HashMap::from([("factor".to_string(), json_to_value(&serde_json::json!(5)))]),
        )
        .await
        .unwrap();

    // dR/da = -factor, dR/dx = 1
    assert_eq!(partials[&("x".to_string(), "a".to_string())][[0]], -5.0);
    assert_eq!(partials[&("x".to_string(), "x".to_string())][[0]], 1.0);

    server.shutdown().await;
}

#[tokio::test]
async fn an_implicit_discrete_default_applies_when_the_client_sends_nothing() {
    let server = common::spawn_implicit(ScaledImplicit::default()).await;
    let mut client = common::connected_implicit_client(&server).await;

    // The declared default of 2 is seeded server-side.
    let outputs = client
        .solve_residuals(&HashMap::from([("a".to_string(), scalar(4.0))]))
        .await
        .unwrap();
    assert_eq!(outputs["x"][[0]], 8.0);

    server.shutdown().await;
}

#[tokio::test]
async fn replacing_an_implicit_discipline_keeps_the_residual_twins() {
    // Regression: `ImplicitServer::new` marked the registry implicit but
    // `set_discipline` did not, so a discipline built with
    // `VariableRegistry::default()` silently lost its residual metadata after a
    // swap. `solve_residuals` kept working, which made the failure quiet.
    let server = spawn_shared_implicit(ScaledImplicit::default()).await;
    let mut client = philote_mdo::client::ImplicitClient::connect(server.endpoint())
        .await
        .expect("connect to the test server");
    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();

    let before = client.base().var_meta().len();

    server
        .server
        .set_discipline(ScaledImplicit::with_offset(0.0))
        .await;

    client.setup().await.unwrap();
    client.get_variable_definitions().await.unwrap();

    assert_eq!(
        client.base().var_meta().len(),
        before,
        "the swapped-in discipline should still declare its residual twin"
    );
    assert!(
        client
            .base()
            .var_meta()
            .iter()
            .any(|m| m.r#type == i32::from(VariableType::KResidual)),
        "no kResidual entry survived the swap"
    );

    // The residuals RPC is what actually breaks when the twin is missing.
    let residuals = client
        .compute_residuals(
            &HashMap::from([("a".to_string(), scalar(4.0))]),
            &HashMap::from([("x".to_string(), scalar(8.0))]),
        )
        .await
        .expect("residuals should decode after a swap");
    assert_eq!(residuals["x"][[0]], 0.0);

    server.shutdown().await;
}
