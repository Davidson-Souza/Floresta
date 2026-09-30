// SPDX-License-Identifier: MIT OR Apache-2.0

use std::hint::black_box;
use std::time::Duration;

use bitcoin::Transaction as BitcoinTransaction;
use bitcoin::consensus::deserialize;
use bitcoin::consensus::serialize;
use bitcoinkernel::PrecomputedTransactionData;
use bitcoinkernel::ScriptPubkey;
use bitcoinkernel::ScriptVerificationFlags;
use bitcoinkernel::SignatureBatch;
use bitcoinkernel::Transaction;
use bitcoinkernel::TxOut;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::SamplingMode;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;

const BATCH_SIZES: [usize; 5] = [1, 8, 32, 128, 512];

const P2PKH_SCRIPT: &str = "76a9144bfbaf6afb76cc5771bc6404810d1cc041a6933988ac";
const P2PKH_TRANSACTION: &str = "02000000013f7cebd65c27431a90bba7f796914fe8cc2ddfc3f2cbd6f7e5f2fc854534da95000000006b483045022100de1ac3bcdfb0332207c4a91f3832bd2c2915840165f876ab47c5f8996b971c3602201c6c053d750fadde599e6f5c4e1963df0f01fc0d97815e8157e3d59fe09ca30d012103699b464d1d8bc9e47d4fb1cdaa89a1c5783d68363c4dbc4b524ed3d857148617feffffff02836d3c01000000001976a914fc25d6d5c94003bf5b0c7b640a248e2c637fcfb088ac7ada8202000000001976a914fbed3d9b11183209a57999d54d59f67c019e756c88ac6acb0700";

const TAPROOT_SCRIPT: &str = "5120339ce7e165e67d93adb3fef88a6d4beed33f01fa876f05a225242b82a631abc0";
const TAPROOT_TRANSACTION: &str = "01000000000101d1f1c1f8cdf6759167b90f52c9ad358a369f95284e841d7a2536cef31c0549580100000000fdffffff020000000000000000316a2f49206c696b65205363686e6f7272207369677320616e6420492063616e6e6f74206c69652e204062697462756734329e06010000000000225120a37c3903c8d0db6512e2b40b0dffa05e5a3ab73603ce8c9c4b7771e5412328f90140a60c383f71bac0ec919b1d7dbc3eb72dd56e7aa99583615564f9f99b8ae4e837b758773a5b2e4c51348854c8389f008e05029db7f464a5ff2e01d5e6e626174affd30a00";
const TAPROOT_AMOUNT: i64 = 88_480;

struct VerificationCase {
    script: ScriptPubkey,
    transaction: Transaction,
    txdata: PrecomputedTransactionData,
    amount: i64,
    flags: ScriptVerificationFlags,
}

fn kernel_transaction(raw: &str) -> Transaction {
    let transaction: BitcoinTransaction = deserialize(&hex::decode(raw).unwrap()).unwrap();
    Transaction::try_from(serialize(&transaction).as_slice()).unwrap()
}

impl VerificationCase {
    fn p2pkh() -> Self {
        let script = ScriptPubkey::try_from(hex::decode(P2PKH_SCRIPT).unwrap().as_slice()).unwrap();
        let transaction = kernel_transaction(P2PKH_TRANSACTION);
        let txdata = PrecomputedTransactionData::new(&transaction, &Vec::<TxOut>::new()).unwrap();
        Self {
            script,
            transaction,
            txdata,
            amount: 0,
            flags: bitcoinkernel::VERIFY_ALL_PRE_TAPROOT,
        }
    }

    fn taproot() -> Self {
        let script =
            ScriptPubkey::try_from(hex::decode(TAPROOT_SCRIPT).unwrap().as_slice()).unwrap();
        let transaction = kernel_transaction(TAPROOT_TRANSACTION);
        let prevout = TxOut::new(&script, TAPROOT_AMOUNT);
        let txdata = PrecomputedTransactionData::new(&transaction, &[prevout]).unwrap();
        Self {
            script,
            transaction,
            txdata,
            amount: TAPROOT_AMOUNT,
            flags: bitcoinkernel::VERIFY_ALL,
        }
    }

    fn verify_kernel(&self, count: usize) {
        for _ in 0..count {
            bitcoinkernel::verify(
                &self.script,
                Some(self.amount),
                &self.transaction,
                0,
                Some(self.flags),
                &self.txdata,
            )
            .unwrap();
        }
    }

    fn collect_deferred(&self, count: usize) -> SignatureBatch {
        let mut batch = SignatureBatch::new().unwrap();
        for _ in 0..count {
            batch
                .verify_deferred(
                    &self.script,
                    Some(self.amount),
                    &self.transaction,
                    0,
                    Some(self.flags),
                    &self.txdata,
                )
                .unwrap();
        }
        batch
    }
}

struct UltraFastCpu(*mut ufsecp_sys::ufsecp_ctx);

impl UltraFastCpu {
    fn new() -> Self {
        let mut context = std::ptr::null_mut();
        let result = unsafe { ufsecp_sys::ufsecp_ctx_create(&mut context) };
        assert_eq!(result, 0, "UltraFastSecp context creation failed");
        assert!(!context.is_null(), "UltraFastSecp returned a null context");
        Self(context)
    }

    fn verify_ecdsa(&mut self, batch: &SignatureBatch) {
        let entries = batch.ecdsa().unwrap();
        let mut rows = Vec::with_capacity(entries.len() * 129);
        for entry in entries {
            rows.extend_from_slice(&entry.message);
            rows.extend_from_slice(&entry.pubkey);
            rows.extend_from_slice(&entry.signature);
        }
        let result = unsafe {
            ufsecp_sys::ufsecp_ecdsa_batch_verify(self.0, rows.as_ptr(), rows.len() / 129)
        };
        assert_eq!(result, 0, "UltraFastSecp ECDSA batch verification failed");
    }

    fn verify_schnorr(&mut self, batch: &SignatureBatch) {
        let entries = batch.schnorr().unwrap();
        let mut rows = Vec::with_capacity(entries.len() * 128);
        for entry in entries {
            rows.extend_from_slice(&entry.pubkey);
            rows.extend_from_slice(&entry.message);
            rows.extend_from_slice(&entry.signature);
        }
        let result = unsafe {
            ufsecp_sys::ufsecp_schnorr_batch_verify(self.0, rows.as_ptr(), rows.len() / 128)
        };
        assert_eq!(result, 0, "UltraFastSecp Schnorr batch verification failed");
    }
}

impl Drop for UltraFastCpu {
    fn drop(&mut self) {
        unsafe { ufsecp_sys::ufsecp_ctx_destroy(self.0) };
    }
}

fn benchmark_ecdsa(c: &mut Criterion) {
    let verification = VerificationCase::p2pkh();
    let mut ultrafast = UltraFastCpu::new();
    verification.verify_kernel(1);
    ultrafast.verify_ecdsa(&verification.collect_deferred(1));

    let mut group = c.benchmark_group("signature_verification/ecdsa");
    group.sample_size(20);
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));
    for count in BATCH_SIZES {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("kernel_immediate", count),
            &count,
            |b, &count| {
                b.iter(|| verification.verify_kernel(black_box(count)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ultrafast_deferred_batch", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let batch = verification.collect_deferred(black_box(count));
                    ultrafast.verify_ecdsa(black_box(&batch));
                });
            },
        );
    }
    group.finish();
}

fn benchmark_schnorr(c: &mut Criterion) {
    let verification = VerificationCase::taproot();
    let mut ultrafast = UltraFastCpu::new();
    verification.verify_kernel(1);
    ultrafast.verify_schnorr(&verification.collect_deferred(1));

    let mut group = c.benchmark_group("signature_verification/schnorr");
    group.sample_size(20);
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));
    for count in BATCH_SIZES {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("kernel_immediate", count),
            &count,
            |b, &count| {
                b.iter(|| verification.verify_kernel(black_box(count)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ultrafast_deferred_batch", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let batch = verification.collect_deferred(black_box(count));
                    ultrafast.verify_schnorr(black_box(&batch));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_ecdsa, benchmark_schnorr);
criterion_main!(benches);
