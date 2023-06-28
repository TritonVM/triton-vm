use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::Measurement;
use criterion::measurement::ValueFormatter;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use twenty_first::shared_math::bfield_codec::BFieldCodec;

use triton_vm::prove_from_source;
use triton_vm::shared_tests::FIBONACCI_SEQUENCE;
use triton_vm::shared_tests::VERIFY_SUDOKU;
use triton_vm::stark::Stark;
use triton_vm::Proof;
use triton_vm::StarkParameters;

#[derive(Debug, Clone, Copy)]
struct ProofSize(f64);

impl Measurement for ProofSize {
    type Intermediate = ();
    type Value = Self;

    fn start(&self) -> Self::Intermediate {}

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        self.to_owned()
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        ProofSize(v1.0 + v2.0)
    }

    fn zero(&self) -> Self::Value {
        ProofSize(0.0)
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        value.0
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &ProofSizeFormatter
    }
}

#[derive(Debug, Clone, Copy)]
struct ProofSizeFormatter;

impl ValueFormatter for ProofSizeFormatter {
    fn scale_values(&self, typical_value: f64, values: &mut [f64]) -> &'static str {
        let bytes_per_bfe = 8.0;
        let divisor_kib = 1024.0;
        let divisor_mib = 1024.0 * divisor_kib;
        let divisor_gib = 1024.0 * divisor_mib;

        let size_in_bytes = typical_value * bytes_per_bfe;
        let size_in_kib = size_in_bytes / divisor_kib;
        let size_in_mib = size_in_bytes / divisor_mib;
        let size_in_gib = size_in_bytes / divisor_gib;

        values.iter_mut().for_each(|v| *v *= bytes_per_bfe);
        let values_in_bytes = values.iter_mut();
        if size_in_kib < 1.0 {
            "bytes"
        } else if size_in_mib < 1.0 {
            values_in_bytes.for_each(|v| *v /= divisor_kib);
            "KiB"
        } else if size_in_gib < 1.0 {
            values_in_bytes.for_each(|v| *v /= divisor_mib);
            "MiB"
        } else {
            values_in_bytes.for_each(|v| *v /= divisor_gib);
            "GiB"
        }
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "bfe/s"
    }

    fn scale_for_machines(&self, values: &mut [f64]) -> &'static str {
        let bytes_per_bfe = 8.0;
        values.iter_mut().for_each(|v| *v *= bytes_per_bfe);
        "bytes"
    }
}

/// The base 2, integer logarithm of the FRI domain length.
fn log_2_fri_domain_length(parameters: &StarkParameters, proof: &Proof) -> u32 {
    let padded_height = proof.padded_height();
    let max_degree = Stark::derive_max_degree(padded_height, parameters.num_trace_randomizers);
    let fri = Stark::derive_fri(parameters, max_degree);
    fri.domain.length.ilog2()
}

fn proof_size(c: &mut Criterion<ProofSize>) {
    let sudoku = [
        1, 2, 3, /**/ 4, 5, 6, /**/ 7, 8, 9, //
        4, 5, 6, /**/ 7, 8, 9, /**/ 1, 2, 3, //
        7, 8, 9, /**/ 1, 2, 3, /**/ 4, 5, 6, //
        /*************************************/
        2, 3, 4, /**/ 5, 6, 7, /**/ 8, 9, 1, //
        5, 6, 7, /**/ 8, 9, 1, /**/ 2, 3, 4, //
        8, 9, 1, /**/ 2, 3, 4, /**/ 5, 6, 7, //
        /*************************************/
        3, 4, 5, /**/ 6, 7, 8, /**/ 9, 1, 2, //
        6, 7, 8, /**/ 9, 1, 2, /**/ 3, 4, 5, //
        9, 1, 2, /**/ 3, 4, 5, /**/ 6, 7, 8, //
    ];

    let (parameters, proof) = prove_from_source("halt", &[], &[]).unwrap();
    let fri_dom_len_halt = log_2_fri_domain_length(&parameters, &proof);
    let bench_id_halt = BenchmarkId::new("halt", fri_dom_len_halt);

    let (parameters, proof) = prove_from_source(FIBONACCI_SEQUENCE, &[100], &[]).unwrap();
    let fri_dom_len_fib_100 = log_2_fri_domain_length(&parameters, &proof);
    let bench_id_fib_100 = BenchmarkId::new("fib_100", fri_dom_len_fib_100);

    let (parameters, proof) = prove_from_source(FIBONACCI_SEQUENCE, &[500], &[]).unwrap();
    let fri_dom_len_fib_500 = log_2_fri_domain_length(&parameters, &proof);
    let bench_id_fib_500 = BenchmarkId::new("fib_500", fri_dom_len_fib_500);

    let (parameters, proof) = prove_from_source(VERIFY_SUDOKU, &sudoku, &[]).unwrap();
    let fri_dom_len_sudoku = log_2_fri_domain_length(&parameters, &proof);
    let bench_id_sudoku = BenchmarkId::new("sudoku", fri_dom_len_sudoku);

    let mut group = c.benchmark_group("proof_size");
    group.bench_function(bench_id_halt, |bencher| {
        bencher.iter_custom(|iters| {
            let mut total_len = 0;
            for _ in 0..iters {
                let (_, proof) = prove_from_source("halt", &[], &[]).unwrap();
                total_len += proof.encode().len();
            }
            ProofSize(total_len as f64)
        })
    });
    group.bench_function(bench_id_fib_100, |bencher| {
        bencher.iter_custom(|iters| {
            let mut total_len = 0;
            for _ in 0..iters {
                let (_, proof) = prove_from_source(FIBONACCI_SEQUENCE, &[100], &[]).unwrap();
                total_len += proof.encode().len();
            }
            ProofSize(total_len as f64)
        })
    });
    group.bench_function(bench_id_fib_500, |bencher| {
        bencher.iter_custom(|iters| {
            let mut total_len = 0;
            for _ in 0..iters {
                let (_, proof) = prove_from_source(FIBONACCI_SEQUENCE, &[500], &[]).unwrap();
                total_len += proof.encode().len();
            }
            ProofSize(total_len as f64)
        })
    });
    group.bench_function(bench_id_sudoku, |bencher| {
        bencher.iter_custom(|iters| {
            let mut total_len = 0;
            for _ in 0..iters {
                let (_, proof) = prove_from_source(VERIFY_SUDOKU, &sudoku, &[]).unwrap();
                total_len += proof.encode().len();
            }
            ProofSize(total_len as f64)
        })
    });
}

fn proof_size_measurements() -> Criterion<ProofSize> {
    Criterion::default()
        .with_measurement(ProofSize(0.0))
        .sample_size(10)
}

criterion_group!(
    name = benches;
    config =  proof_size_measurements();
    targets = proof_size
);
criterion_main!(benches);
