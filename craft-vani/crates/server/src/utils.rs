use anyhow::Result;
use candle::Device;
use candle::utils::{cuda_is_available, metal_is_available};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else{ 
        Ok(Device::new_cuda(0)?)
    } 
}
