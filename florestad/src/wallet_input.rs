//!
//! Handles different inputs, try to make sense out of it and store a sane descriptor at the end

use std::str::FromStr;

use bitcoin::Address;
use miniscript::Descriptor;
use miniscript::DescriptorPublicKey;

use floresta_errors::florestad::commom::FlorestadError;


pub mod extended_pub_key {
    use bitcoin::bip32::Xpub;
    use floresta_errors::florestad::slip123::Slip32Errors;

    use crate::slip132;

    pub fn from_str(s: &str) -> Result<Xpub, Slip32Errors> {
        slip132::FromSlip132::from_slip132_str(s)
    }
}

fn parse_xpubs(xpubs: &[String]) -> Result<Vec<Descriptor<DescriptorPublicKey>>, FlorestadError> {
    let mut descriptors = Vec::new();
    for key in xpubs {
        // Parses the descriptor and get an external and change descriptors
        let xpub = extended_pub_key::from_str(key.as_str()).map_err(|e| {
            log::error!("Invalid xpub provided: {key} \nReason: {e:?}");
            e
        })?;
        let main_desc = format!("wpkh({xpub}/0/*)");
        let change_desc = format!("wpkh({xpub}/1/*)");
        descriptors.push(Descriptor::<DescriptorPublicKey>::from_str(&main_desc)?);
        descriptors.push(Descriptor::<DescriptorPublicKey>::from_str(&change_desc)?);
    }
    Ok(descriptors)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InitialWalletSetup {
    pub(crate) descriptors: Vec<Descriptor<DescriptorPublicKey>>,
    pub(crate) addresses: Vec<Address>,
}

impl InitialWalletSetup {
    pub(crate) fn build(
        xpubs: &[String],
        initial_descriptors: &[String],
        addresses: &[String],
        network: bitcoin::Network,
        addresses_per_descriptor: u32,
    ) -> Result<Self, FlorestadError> {
        let mut descriptors = parse_xpubs(xpubs)?;
        descriptors.extend(parse_descriptors(initial_descriptors)?);
        descriptors.sort();
        descriptors.dedup();
        let mut addresses = addresses
            .iter()
            .flat_map(|address| match Address::from_str(address) {
                Ok(address) => Ok(address.require_network(network)),
                Err(e) => {
                    log::error!("Invalid address provided: {address} \nReason: {e:?}");
                    Err(e)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        addresses.extend(descriptors.iter().flat_map(|descriptor| {
            (0..addresses_per_descriptor).map(|index| {
                descriptor
                    .at_derivation_index(index)
                    .expect("Error while deriving address")
                    .address(network)
                    .expect("Error while deriving address. Is this an active descriptor?")
            })
        }));
        addresses.sort();
        addresses.dedup();
        Ok(Self {
            descriptors,
            addresses,
        })
    }
}

pub fn parse_descriptors(
    descriptors: &[String],
) -> Result<Vec<Descriptor<DescriptorPublicKey>>, FlorestadError> {
    let descriptors = descriptors
        .iter()
        .map(|descriptor| {
            let descriptor = Descriptor::<DescriptorPublicKey>::from_str(descriptor.as_str())?;
            descriptor.sanity_check()?;
            descriptor.into_single_descriptors()
        })
        .collect::<Result<Vec<Vec<_>>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok(descriptors)
}

#[cfg(test)]
pub mod test {
    use bitcoin::bip32::ChildNumber;
    use bitcoin::secp256k1::Secp256k1;
    use bitcoin::Network;

    use super::*;

    #[test]
    fn test_xpub_parsing() {
        // Test cases from https://github.com/satoshilabs/slips/blob/master/slip-0132.md
        const XPUB: &str = "xpub6BosfCnifzxcFwrSzQiqu2DBVTshkCXacvNsWGYJVVhhawA7d4R5WSWGFNbi8Aw6ZRc1brxMyWMzG3DSSSSoekkudhUd9yLb6qx39T9nMdj";
        const YPUB: &str = "ypub6Ww3ibxVfGzLrAH1PNcjyAWenMTbbAosGNB6VvmSEgytSER9azLDWCxoJwW7Ke7icmizBMXrzBx9979FfaHxHcrArf3zbeJJJUZPf663zsP";
        const ZPUB: &str = "zpub6rFR7y4Q2AijBEqTUquhVz398htDFrtymD9xYYfG1m4wAcvPhXNfE3EfH1r1ADqtfSdVCToUG868RvUUkgDKf31mGDtKsAYz2oz2AGutZYs";

        let secp = Secp256k1::new();

        let xpub: bitcoin::bip32::Xpub = super::extended_pub_key::from_str(XPUB)
            .expect("Parsing failed")
            .ckd_pub(&secp, ChildNumber::Normal { index: 0 })
            .and_then(|key| key.ckd_pub(&secp, ChildNumber::Normal { index: 0 }))
            .unwrap();
        let ypub = super::extended_pub_key::from_str(YPUB)
            .expect("Parsing failed")
            .ckd_pub(&secp, ChildNumber::Normal { index: 0 })
            .and_then(|key| key.ckd_pub(&secp, ChildNumber::Normal { index: 0 }))
            .unwrap();
        let zpub = super::extended_pub_key::from_str(ZPUB)
            .expect("Parsing failed")
            .ckd_pub(&secp, ChildNumber::Normal { index: 0 })
            .and_then(|key| key.ckd_pub(&secp, ChildNumber::Normal { index: 0 }))
            .unwrap();
        // Old p2pkh
        assert_eq!(
            Address::p2pkh(&xpub.to_pub(), bitcoin::Network::Bitcoin)
                .to_string()
                .as_str(),
            "1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA"
        );
        // p2wpkh-p2pkh
        let script = Address::p2wpkh(&ypub.to_pub(), bitcoin::Network::Bitcoin)
            .unwrap()
            .script_pubkey();

        assert_eq!(
            Address::p2sh(&script, Network::Bitcoin)
                .unwrap()
                .to_string()
                .as_str(),
            "37VucYSaXLCAsxYyAPfbSi9eh4iEcbShgf"
        );

        // p2wpkh
        assert_eq!(
            Address::p2wpkh(&zpub.to_pub(), bitcoin::Network::Bitcoin)
                .unwrap()
                .to_string()
                .as_str(),
            "bc1qcr8te4kr609gcawutmrza0j4xv80jy8z306fyu"
        )
    }

    #[test]
    fn test_descriptor_parsing() {
        // singlesig
        assert_eq!(
            parse_descriptors(&[
                "wpkh([a5b13c0e/84h/0h/0h]xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/<0;1>/*)#n8sgapuv".to_owned()
            ]).unwrap(),
            parse_descriptors(&[
                "wpkh([a5b13c0e/84'/0'/0']xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/0/*)#wg8dh3s7".to_owned(),
                "wpkh([a5b13c0e/84'/0'/0']xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/1/*)#luzv2yqx".to_owned()
            ]).unwrap()
        );
        // multisig
        assert_eq!(
            parse_descriptors(&[
                "wsh(sortedmulti(1,[6f826a6a/48h/0h/0h/2h]xpub6DsY48BAsvEMTRPbeSTu9jZXqEsTKr5T86WbRbXHp2gEVCNR3hALnMorFawVwnnHMMfjbyY8We9B4beh1fxqhcv6kgSeLgQxeXDqv3DaW7m/<0;1>/*,[a5b13c0e/48h/0h/0h/2h]xpub6Eqj1Hj3RezebC6cKiYYN2sAc1Wu33BWoaafnNgAbQwDkJdy7aXCYCmaMzb8rCpmh919UsehyV5Ywjo62hG4R2G2PGv4uqEDTUhYQw26BDJ/<0;1>/*))#nykmcu2v".to_owned()
            ]).unwrap(),
            parse_descriptors(&[
                "wsh(sortedmulti(1,[6f826a6a/48'/0'/0'/2']xpub6DsY48BAsvEMTRPbeSTu9jZXqEsTKr5T86WbRbXHp2gEVCNR3hALnMorFawVwnnHMMfjbyY8We9B4beh1fxqhcv6kgSeLgQxeXDqv3DaW7m/0/*,[a5b13c0e/48'/0'/0'/2']xpub6Eqj1Hj3RezebC6cKiYYN2sAc1Wu33BWoaafnNgAbQwDkJdy7aXCYCmaMzb8rCpmh919UsehyV5Ywjo62hG4R2G2PGv4uqEDTUhYQw26BDJ/0/*))#sw68w95x".to_owned(),
                "wsh(sortedmulti(1,[6f826a6a/48'/0'/0'/2']xpub6DsY48BAsvEMTRPbeSTu9jZXqEsTKr5T86WbRbXHp2gEVCNR3hALnMorFawVwnnHMMfjbyY8We9B4beh1fxqhcv6kgSeLgQxeXDqv3DaW7m/1/*,[a5b13c0e/48'/0'/0'/2']xpub6Eqj1Hj3RezebC6cKiYYN2sAc1Wu33BWoaafnNgAbQwDkJdy7aXCYCmaMzb8rCpmh919UsehyV5Ywjo62hG4R2G2PGv4uqEDTUhYQw26BDJ/1/*))#fafrqkpn".to_owned()
            ]).unwrap()
        );
    }

    #[test]
    fn test_initial_wallet_build() {
        use pretty_assertions::assert_eq;
        let addresses_per_descriptor = 1;
        let network = Network::Bitcoin;
        // Build wallet from xpub (in this case a zpub from slip132 standard)
        let w1_xpub = InitialWalletSetup::build(&[
            "zpub6qvVf5mN7DH14wr7oZS7xL6cpcFPDmxFEHBk18YpUF1qnroE1yGfW83eafbbi23dzRk7jrVXeJFMyCo3urmQpwkXtVnRmGmaJ3qVvdwx4mB".to_owned()
        ], &[], &[], network, addresses_per_descriptor).unwrap();
        // Build same wallet from output descriptor
        let w1_descriptor = InitialWalletSetup::build(&[], &[
            "wpkh(xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/<0;1>/*)".to_owned()
        ], &[], network, addresses_per_descriptor).unwrap();
        // Using both methods the result should be the same
        assert_eq!(w1_xpub, w1_descriptor);
        // Both normal receiving descriptor and change descriptor should be present
        assert_eq!(
            w1_descriptor.descriptors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            vec![
                "wpkh(xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/0/*)#qua4l7ct",
                "wpkh(xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/1/*)#3gc5ztgn"
            ]
        );
        // Receiving and change addresses
        let addresses = vec![
            "bc1q88guum89mxwszau37m3y4p24renwlwgtkscl6x".to_owned(),
            "bc1q24629yendf7q0dxnw362dqccn52vuz9s0z59hr".to_owned(),
        ];
        assert_eq!(
            w1_descriptor
                .addresses
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            addresses
        );
        // We can build from these addresses
        let w1_addresses =
            InitialWalletSetup::build(&[], &[], &addresses, network, addresses_per_descriptor)
                .unwrap();
        // And the result will be the same as from xpub/descriptor
        assert_eq!(w1_descriptor.addresses, w1_addresses.addresses);
        // We can also build from xpub, descriptor and addresses, at same time
        let w1_all =
            InitialWalletSetup::build(&[
                "zpub6qvVf5mN7DH14wr7oZS7xL6cpcFPDmxFEHBk18YpUF1qnroE1yGfW83eafbbi23dzRk7jrVXeJFMyCo3urmQpwkXtVnRmGmaJ3qVvdwx4mB".to_owned()
            ], &[
                "wpkh(xpub6CFy3kRXorC3NMTt8qrsY9ucUfxVLXyFQ49JSLm3iEG5gfAmWewYFzjNYFgRiCjoB9WWEuJQiyYGCdZvUTwPEUPL9pPabT8bkbiD9Po47XG/<0;1>/*)".to_owned()
            ], &addresses, network, addresses_per_descriptor).unwrap();
        // And the result should be the same, no duplication will happen
        assert_eq!(w1_descriptor, w1_all);
    }

    #[test]
    fn test_initial_wallet_build_multisig_testnet() {
        use pretty_assertions::assert_eq;
        let addresses_per_descriptor = 1;
        let network = Network::Testnet;
        let w1_descriptor = InitialWalletSetup::build(&[], &[
            "wsh(sortedmulti(1,[54ff5a12/48h/1h/0h/2h]tpubDDw6pwZA3hYxcSN32q7a5ynsKmWr4BbkBNHydHPKkM4BZwUfiK7tQ26h7USm8kA1E2FvCy7f7Er7QXKF8RNptATywydARtzgrxuPDwyYv4x/<0;1>/*,[bcf969c0/48h/1h/0h/2h]tpubDEFdgZdCPgQBTNtGj4h6AehK79Jm4LH54JrYBJjAtHMLEAth7LuY87awx9ZMiCURFzFWhxToRJK6xp39aqeJWrG5nuW3eBnXeMJcvDeDxfp/<0;1>/*))#fuw35j0q".to_owned()
        ], &[], network, addresses_per_descriptor).unwrap();
        let addresses = vec![
            "tb1q2eeqw57e7pmrh5w3wkrshctx2qk80vf4mu7l7ek3ne4hg3lmcrnqcwejgj".to_owned(),
            "tb1q6dpyc3jyqelgfwksedef0k2244rcg4gf6wvqm463lk907es2m08qnrfky7".to_owned(),
        ];
        assert_eq!(
            w1_descriptor
                .addresses
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            addresses
        );
    }
}
