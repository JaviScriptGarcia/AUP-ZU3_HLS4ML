# build_accel.tcl
#
# Synthesizes the BloodMNIST MobileNet-mini AXI-Stream + AXI-Lite wrapper
# around the hls4ml-generated `myproject` core, producing a Vivado IP
# catalog entry with ports `s_axis_in`, `s_axis_out` and `s_axi_CTRL`.
#
# Run with:
#   cd 03.hls && vitis_hls -f build_accel.tcl
#
# Input:  the hls4ml C++ project at ../02.hls4ml/output/myproject_wbc_prj
# Output: ./ip/ with the packaged IP.

set hls4ml_prj "../02.hls4ml/output/myproject_wbc_prj"
set firmware   "${hls4ml_prj}/firmware"

open_project -reset myproject_accel_prj

set_top myproject_wbc_accel

# Core hls4ml-generated files
add_files ${firmware}/myproject.cpp        -cflags "-std=c++14 -I${firmware} -I."
add_files ${firmware}/defines.h
add_files ${firmware}/parameters.h
add_files ${firmware}/nnet_utils
add_files ${firmware}/weights

# Our AXI wrapper
add_files myproject_wbc_accel.cpp          -cflags "-std=c++14 -I${firmware} -I."
add_files myproject_wbc_accel.h

open_solution -reset "solution1"

set_part {xczu3eg-sfvc784-2-e}

create_clock -period 10 -name default
config_compile  -name_max_length 80
config_schedule -enable_dsp_full_reg=false

# HLS synthesis
csynth_design

# Export IP to ./ip/
config_export -format ip_catalog -rtl verilog -output ./ip
export_design  -format ip_catalog -rtl verilog -output ./ip

exit
