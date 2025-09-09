use std::str::FromStr;

use bitcoin::consensus::Decodable;
use bitcoin::hashes::Hash;
use bitcoin::Block;
use bitcoin::BlockHash;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BatchSize;
use criterion::Criterion;
use floresta_chain::proof_util::process_proof;
use floresta_chain::proof_util::UtreexoLeafError;
use floresta_chain::UData;
use rustreexo::accumulator::node_hash::BitcoinNodeHash;
use rustreexo::accumulator::pollard::Pollard;
use rustreexo::accumulator::proof::Proof;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Serialize, Deserialize)]
struct TestVector {
    provedathash: BlockHash,
    proofhashes: Vec<String>,
    prooftargets: Vec<u64>,
    hashesproven: Vec<String>,
    hex: String,
}

fn get_block_pollard() -> Pollard<BitcoinNodeHash> {
    let roots = vec![
        "65f33fd7e90f5b46e32b3ddb02f2aa8f0b304ff0e0388e62f59115b85d9b9816",
        "50a08c39b8be6080db8f74c893458b3f47a808b6a8a0e69ba62a9e2178e615d9",
        "0889041e784acfd32d872b10d3342c63d73148dd877cb7f519f764f5763dbeee",
        "dc1529cffa34f4c4516624a1676572f5574671ce75736cd42f8de474146aed6c",
        "90653f07dc08a70bec35673ed1d0eea5c3fba1f1580ae892f9f3af7c9a34e14a",
        "a6b780f74e8481f7fb9a38bdd61f1b330d0cf7845db5983edb195d535f8d5035",
        "7798a5ba7230665b5e96369da9a7c423b81af62771b7a046f304ac0ce5aca078",
        "3b8142862972ceb26e66b18a7ba4d9041f9826185d008a2db9f024619e42954e",
        "47c4f15f76acabe66522393eb3608f3f65a43ea17cc7accd003e845a78534406",
        "e1a35148e6cf9ce76d12606925b44f7daed2850b37ab0ce476e1639ea37d708c",
        "5a2c57583c902566c1ee2f0a8a39818cdb75116058770fa012a5e60ded22c7d7",
        "ef1658d2f1e72935cc87d616726269b6d19bcc08ccffdf6c9ddb3079fb2d333c",
        "cdc1429b72983fc79ece6dab6cd1797df8c9a9001cf1b3054a88c26e685cdb9f",
        "b76771d2fff7b9e67daa102f423106fe074c5c34da7671851b707b7918f2e198",
        "16040bf57c57cfc7da4893ce81158ab92a63f4d8d46944b2a630aaf2fc22c839",
        "832ab588cf8f8aa232f88f48c5edf1133e548100262132597f3a06699995fd7a",
    ]
    .iter()
    .map(|h| BitcoinNodeHash::from_str(h).unwrap())
    .collect();

    let numleaves = 2923703213;
    Pollard::from_roots(roots, numleaves)
}

fn get_udata() -> UData {
    let mut hex = include_str!("../testdata/bench/block_proof.txt");
    hex = hex.trim();
    let bytes = hex::decode(hex).unwrap();

    UData::consensus_decode(&mut bytes.as_slice()).unwrap()
}

fn get_block() -> Block {
    let mut hex = include_str!("../testdata/bench/block.txt");
    hex = hex.trim();
    let bytes = hex::decode(hex).unwrap();

    Block::consensus_decode(&mut bytes.as_slice()).unwrap()
}

fn get_txout_proofs() -> Vec<Proof<BitcoinNodeHash>> {
    let test_vectors: TestVector =
        serde_json::from_str(include_str!("../testdata/bench/proof.json")).unwrap();
    let proof = Proof {
        hashes: test_vectors
            .proofhashes
            .iter()
            .map(|h| BitcoinNodeHash::from_str(h).unwrap())
            .collect(),
        targets: test_vectors.prooftargets.clone(),
    };

    vec![proof]
}

fn bench_ingest_one(c: &mut Criterion) {
    let setup_function = || {
        let pollard = get_block_pollard();
        let proofs = get_txout_proofs();
        let del_hashes: Vec<_> = proofs[0].hashes.clone();
        (pollard, proofs, del_hashes)
    };

    c.bench_function("bench_ingest_one", |b| {
        b.iter_batched(
            setup_function,
            |(pollard, proofs, del_hashes)| {
                let mut p = pollard;
                p.ingest_proof(proofs[0].clone(), &del_hashes, &proofs[0].targets)
                    .unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_modify_acc(c: &mut Criterion) {
    let block_hashes = include_str!("../testdata/bench/block_hashes.txt");
    let block_hashes: Vec<BlockHash> = block_hashes
        .lines()
        .map(|line| BlockHash::from_str(line).unwrap())
        .collect();

    let setup_function = || {
        let mut pollard = get_block_pollard();
        let proofs = get_txout_proofs();
        let del_hashes: Vec<_> = proofs[0].hashes.clone();
        pollard
            .ingest_proof(proofs[0].clone(), &del_hashes, &proofs[0].targets)
            .unwrap();
        let block = get_block();
        let udata = get_udata();
        let (proof, del_hashes, _) = process_proof(
            &udata,
            &block.txdata,
            913909,
            |height| -> Result<BlockHash, UtreexoLeafError> {
                Ok(*block_hashes.get(height as usize).unwrap())
            },
        )
        .unwrap();

        let del_hashes: Vec<BitcoinNodeHash> = del_hashes
            .iter()
            .map(|h| BitcoinNodeHash::from(h.to_byte_array()))
            .collect();

        (pollard, proof, del_hashes)
    };

    c.bench_function("modify_acc", |b| {
        b.iter_batched(
            setup_function,
            |(mut pollard, proof, del_hashes)| {
                pollard.modify(&[], &del_hashes, proof).unwrap();
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_ingest_one, bench_modify_acc);
criterion_main!(benches);
