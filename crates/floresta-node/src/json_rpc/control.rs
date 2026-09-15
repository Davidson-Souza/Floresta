// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;

use super::res::jsonrpc_interface::JsonRpcError;
use super::server::RpcChain;
use super::server::RpcImpl;

impl<Blockchain: RpcChain> RpcImpl<Blockchain> {
    pub(super) fn get_memory_info(&self, mode: &str) -> Result<GetMemInfoRes, JsonRpcError> {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        match mode {
            "stats" => {
                let info = unsafe { libc::mallinfo() };

                let stats = GetMemInfoStats {
                    locked: MemInfoLocked {
                        used: info.uordblks as u64,
                        free: info.fordblks as u64,
                        total: (info.uordblks + info.fordblks) as u64,
                        locked: info.hblkhd as u64,
                        chunks_used: info.ordblks as u64,
                        chunks_free: info.smblks as u64,
                    },
                };

                Ok(GetMemInfoRes::Stats(stats))
            }

            "mallocinfo" => {
                // A XML with the allocator statistics
                let info = unsafe { libc::mallinfo() };
                let info_str = format!(
                    "<malloc version=\"2.0\"><heap nr=\"1\"><allocated>{}</allocated><free>{}</free><total>{}</total><locked>{}</locked><chunks nr=\"{}\"><used>{}</used><free>{}</free></chunks></heap></malloc>",
                    info.hblkhd,
                    info.uordblks,
                    info.fordblks,
                    info.uordblks + info.fordblks,
                    info.hblkhd,
                    info.ordblks,
                    info.smblks,
                );

                Ok(GetMemInfoRes::MallocInfo(info_str))
            }

            _ => Err(JsonRpcError::InvalidMemInfoMode),
        }

        #[cfg(target_os = "macos")]
        match mode {
            "stats" => {
                let mut info: libc::malloc_statistics_t = unsafe { std::mem::zeroed() };
                unsafe {
                    libc::malloc_zone_statistics(std::ptr::null_mut(), &mut info);
                }

                let stats = GetMemInfoStats {
                    locked: MemInfoLocked {
                        used: info.size_in_use as u64,
                        free: info.size_allocated.saturating_sub(info.size_in_use) as u64,
                        total: info.size_allocated as u64,
                        locked: info.size_allocated as u64,
                        chunks_used: info.blocks_in_use as u64,
                        chunks_free: 0, // Not available on MacOS
                    },
                };

                Ok(GetMemInfoRes::Stats(stats))
            }
            "mallocinfo" => {
                // A XML with the allocator statistics
                let mut info: libc::malloc_statistics_t = unsafe { std::mem::zeroed() };
                unsafe {
                    libc::malloc_zone_statistics(std::ptr::null_mut(), &mut info);
                }

                let info_str = format!(
                    "<malloc version=\"2.0\"><heap nr=\"1\"><allocated>{}</allocated><free>{}</free><total>{}</total><locked>{}</locked><chunks nr=\"{}\"><used>{}</used><free>{}</free></chunks></heap></malloc>",
                    info.size_allocated,
                    info.size_in_use,
                    info.size_allocated - info.size_in_use,
                    info.size_allocated,
                    info.size_allocated,
                    info.blocks_in_use,
                    0
                );

                Ok(GetMemInfoRes::MallocInfo(info_str))
            }
            _ => Err(JsonRpcError::InvalidMemInfoMode),
        }

        #[cfg(not(any(all(target_os = "linux", target_env = "gnu"), target_os = "macos")))]
        // Return zeroed stats on targets without a supported allocator API.
        match mode {
            "stats" => Ok(GetMemInfoRes::Stats(GetMemInfoStats::default())),
            "mallocinfo" => Ok(GetMemInfoRes::MallocInfo(String::new())),
            _ => Err(JsonRpcError::InvalidMemInfoMode),
        }
    }

    pub(super) async fn get_rpc_info(&self) -> Result<GetRpcInfoRes, JsonRpcError> {
        let active_commands = self
            .inflight
            .read()
            .await
            .values()
            .map(|req| ActiveCommand {
                method: req.method.clone(),
                duration: req.when.elapsed().as_micros() as u64,
            })
            .collect();

        Ok(GetRpcInfoRes {
            active_commands,
            logpath: self.log_path.clone(),
        })
    }

    // help
    // logging

    // stop
    pub(super) async fn stop(&self) -> Result<&str, JsonRpcError> {
        *self.kill_signal.write().await = true;

        Ok("Floresta stopping")
    }

    // uptime
    pub(super) fn uptime(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GetMemInfoStats {
    locked: MemInfoLocked,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MemInfoLocked {
    used: u64,
    free: u64,
    total: u64,
    locked: u64,
    chunks_used: u64,
    chunks_free: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GetMemInfoRes {
    Stats(GetMemInfoStats),
    MallocInfo(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActiveCommand {
    method: String,
    duration: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetRpcInfoRes {
    active_commands: Vec<ActiveCommand>,
    logpath: PathBuf,
}
