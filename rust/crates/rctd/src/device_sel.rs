use rctd_core::device_cpu;
use rctd_core::RctdDevice;
#[cfg(feature = "wgpu")]
use rctd_core::default_device;

#[cfg(not(feature = "wgpu"))]
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum ComputeDevice {
    #[default]
    Cpu,
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum ComputeDevice {
    #[default]
    Cpu,
    Wgpu,
}

pub fn resolve(which: ComputeDevice) -> RctdDevice {
    match which {
        ComputeDevice::Cpu => device_cpu(),
        #[cfg(feature = "wgpu")]
        ComputeDevice::Wgpu => default_device(),
    }
}
