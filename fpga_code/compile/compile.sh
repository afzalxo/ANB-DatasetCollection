#!/bin/bash

archjsons=("/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json" "/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json")
targets=("zcu102" "vck190")
# ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
# TARGET=zcu102
# ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
# TARGET=vck190

JOB_ID=$1
MODEL_V=$2
BUILD=$3

for i in "${!targets[@]}"
do
	TARGET=${targets[$i]}
	ARCH=${archjsons[$i]}
	echo "-----------------------------------------"
	echo "COMPILING MODEL FOR ${TARGET}.."
	echo "-----------------------------------------"

	compile() {
	  vai_c_xir \
	  --xmodel      ${BUILD}/quant_model/AccelNASBenchNet_int.xmodel \
	  --arch        $ARCH \
	  --net_name    model_${JOB_ID}_${MODEL_V}_${TARGET} \
	  --output_dir  ${BUILD}/compiled_model
	}

	compile

	echo "-----------------------------------------"
	echo "MODEL COMPILED"
	echo "-----------------------------------------"
done
