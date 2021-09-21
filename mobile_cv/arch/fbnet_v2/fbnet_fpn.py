#!/usr/bin/env python3

import copy
from typing import Dict, List, Optional, Tuple

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.common.misc.registry as registry
import torch
import torch.nn as nn
from mobile_cv.arch.fbnet_v2.fbnet_builder import FBNetBuilder


STAGE_COMBINERS_REGISTRY = registry.Registry("stage_combiners")


class FBNetFPNBuilder:
    def __init__(self, builder: FBNetBuilder):
        """
        Builder to create feature pyramid network (FPN) with the shape:

        Input0   Input1   Input2   Input3
          |        |        |        |
        Stage0   Stage1   Stage5   Stage6
          |        |        |        |
           Combiner    |---> Combiner    |--->
              |        |        |        |
            Stage2     |      Stage7     |
              |        |        |        |
              | ---> Stage4     | ---> Stage9
            Stage3            Stage8
              |                 |
            Output0           Output1

        This builder is used to create different variants of feature pyramid
        networks (FPNs). A typical network usually consists of several paths,
        with each path corresponding to a fixed spatial resolution, and cross-path
        connections to fuse information from different paths (i.e., spatial
        resolutions). Each path takes two inputs to accommedate the residual
        connections proposed in EfficientDet. Cross-path connections, e.g.,
        Stage 4 and 9, conduct down-sampling or up-sampling and currently only a
        uni-directional information flow is supported. The combiner is by default
        an add operation but can be specified when building the model.

        In order to create the network, create this builder and then, call build_model
        by specifying the number of spatial resolutions and a valid arch_def:
            fbnet_builder = FBNetBuilder()
            fbnet_fpn_builder = FBNetFPNBuilder(fbnet_builder)
            model = fbnet_fpn_builder.build_model(num_resolutions, arch_def)

        See build_model to get an example of valid arch_def.
        Args:
            Builder stores general information about blocks (e.g., batch norm type)
        """
        self.builder = builder
        self.blocks = None
        self.blocks_out_dims = None
        self.num_stages = None
        self.num_resolutions = None
        self.input_channels = None
        self.combiner_path = None

    def build_model(self, input_channels: List[int], arch_def: Dict) -> nn.Module:
        """Create an FPN specified arch_def

        Arch_def should contain the following information:
            arch_def: {
                "stages": [
                    [(op0, c, n, s, ...), (op1, c, n, s, ...)],
                    [(op2, c, n, s, ...)],
                    ...
                    [(opN, c, n, s, ...)]
                ],
                "stage_combiners": Optional[List[str]],
                "combiner_path": Optional[str],
            }
        Note that stages are listed in order by stage:
            "stages": [[Stage0 ops], [Stage1 ops], ..., [StageN ops]]

        Example:
            arch_def = {
                "stages": [],
                "stage_combiners": ["add"],
            }
            fbnet_builder = FBNetBuilder()
            fbnet_fpn_builder = FBNetHRBuilder(fbnet_builder)
            model = fbnet_fpn_builder.build_model(arch_def)

        The resulting model has the structure:
        """
        self._verify_init(input_channels, arch_def)
        self.build_stages(self.blocks)
        self.build_stage_combiners(
            # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute
            #  `stage_combiner_num_inputs`.
            self.stage_combiner_num_inputs,
            arch_def.get("stage_combiners", None),
        )
        return FBNetFPN(
            # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute `stages`.
            self.stages,
            # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute `stage_combiners`.
            self.stage_combiners,
            # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute `output_channels`.
            self.output_channels,
            self.combiner_path,
        )

    def _verify_init(self, input_channels: List[int], arch_def: Dict) -> None:
        """Check that arch_def is a valid FBNetFPN definition

        Valid definition properties:
            * input_channels specified
            * stages specified
            * num_resolutions = len(input_channels) // 2
            * stage combiners = num_resolutions
            * num_stages = 4 * num_resolutions + num_resolutions - 1
            * inputs to combiners have specific number of channels
                * add: channels should be the same
                * mul: channels should be the same or 1

        Note that this function sets the following parameters:
            * input_channels, num_resolutions, num_stages_per_resolution (by
              default = 4), blocks, blocks_out_dims, combiner_path
        """
        # TODO: do we want to put "input_channels" into arch_def
        assert isinstance(input_channels, list) and len(input_channels) > 0
        self.input_channels = input_channels
        self.num_resolutions = len(self.input_channels) // 2
        # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute
        #  `num_stages_per_resolution`.
        self.num_stages_per_resolution = 5

        assert "stages" in arch_def
        # self.num_stages = len(arch_def["stages"])
        self.num_stages = self.num_resolutions * self.num_stages_per_resolution - 1
        assert (
            # num_stages == num_resolutions * num_stages_per_resolution - 1
            len(arch_def["stages"])
            >= self.num_stages
        ), (
            f"FBNet FPN requires 4 stages per spatial resolution and in "
            f"total {self.num_resolutions - 1} stages for cross-resolution connections"
        )

        default_combiners = ["add"] * self.num_resolutions
        stage_combiners = arch_def.get("stage_combiners", default_combiners)
        assert isinstance(stage_combiners, list)
        assert len(stage_combiners) == self.num_resolutions

        # iterate over stages and check that the inputs to the combiners
        # have the same number of channels or 1
        self.blocks = fbnet_builder.unify_arch_def_blocks(arch_def["stages"])
        self.blocks_out_dims = fbnet_builder.get_stages_dim_out(self.blocks)
        # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute
        #  `stage_combiner_num_inputs`.
        self.stage_combiner_num_inputs = [0] * self.num_resolutions
        for i in range(self.num_resolutions):
            input_chdepths = []

            inputA_stage = i * self.num_stages_per_resolution
            if self.blocks[inputA_stage]["block_op"] != "noop":
                input_chdepths.append(self.blocks_out_dims[inputA_stage])

            inputB_stage = i * self.num_stages_per_resolution + 1
            if self.blocks[inputB_stage]["block_op"] != "noop":
                input_chdepths.append(self.blocks_out_dims[inputB_stage])

            if i > 0:
                inputC_stage = i * self.num_stages_per_resolution - 1
                if self.blocks[inputC_stage]["block_op"] != "noop":
                    input_chdepths.append(self.blocks_out_dims[inputC_stage])

            stage_combiner = stage_combiners[i]
            if stage_combiner == "add":
                assert (
                    len(set(input_chdepths)) == 1
                ), f"Trying to add features with different chls: {input_chdepths}"
                self.stage_combiner_num_inputs[i] = len(input_chdepths)
            else:
                raise NotImplementedError

        # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute `output_channels`.
        self.output_channels = []
        for i in range(self.num_resolutions):
            self.output_channels.append(self.blocks_out_dims[i * 5 + 3])

        self.combiner_path = arch_def.get("combiner_path", "high_res")

        # always order the output from high res to low res
        if self.combiner_path == "low_res":
            self.output_channels = self.output_channels[::-1]

    def get_prev_stages(
        self, stage_id: int
    ) -> Tuple[int, Optional[int], Optional[int]]:
        """Returns the id of the stage before stage_id

        Since this network has different paths, we have this function which
        returns:
            if stage has a single input:
                if stage input is fpn input (i.e., i == 0 or i == 1):
                    (input id, None, None)
                if stage input is not fpn input (i.e., i == 3 or i == 4):
                    (prev stage, None, None)
            if stage has 3 input:
                (prev same-res stage 1, prev same-res stage 2, prev cross-res stage)
        Inputs: (int) stage_id
        Return: (tuple(int)) previous stage_ids
        """
        # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute
        #  `num_stages_per_resolution`.
        i = stage_id % self.num_stages_per_resolution
        j = stage_id // self.num_stages_per_resolution
        if i == 0 or i == 1:
            return (j * 2 + i, None, None)
        elif i == 2 and j == 0:
            return (
                self.num_stages_per_resolution * j,
                self.num_stages_per_resolution * j + 1,
                None,
            )
        elif i == 2 and j > 0:
            return (
                self.num_stages_per_resolution * j,
                self.num_stages_per_resolution * j + 1,
                self.num_stages_per_resolution * j - 1,
            )
        elif i == 3 or i == 4:
            return (self.num_stages_per_resolution * j + 2, None, None)
        else:
            raise ValueError

    def get_prev_stages_output_dims(
        self, stage_id: int
    ) -> Tuple[int, Optional[int], Optional[int]]:
        """Returns the dimensions of the previous stages

        Stage can have 1 or 2 or 3 previous stages. Returns None
        in the tuple if corresponding input stage is None
        Input: (dict) arch_def
               (int) stage id
        Return: (tuple(int)) dimensions of the previous stages
        """
        dims_in = [None, None, None]
        prev_stages = self.get_prev_stages(stage_id)
        for i in range(len(dims_in)):
            if prev_stages[i] is not None:
                dims_in[i] = self.blocks_out_dims[prev_stages[i]]
        # pyre-fixme[7]: Expected `Tuple[int, Optional[int], Optional[int]]` but got
        #  `Tuple[None, ...]`.
        return tuple(dims_in)

    def build_stages(self, blocks: List) -> None:
        """ """
        # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute `stages`.
        self.stages = nn.ModuleList(
            [
                # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute
                #  `num_stages_per_resolution`.
                nn.ModuleList([None] * self.num_stages_per_resolution)
                for _ in range(self.num_resolutions)
            ]
        )
        for k in range(self.num_stages):
            i = k % self.num_stages_per_resolution
            j = k // self.num_stages_per_resolution

            if i == 0 or i == 1:
                dim_in = self.input_channels[j * 2 + i]
            else:
                dims_in = self.get_prev_stages_output_dims(k)
                dim_in = dims_in[0]

            self.stages[j][i] = self.builder.build_blocks(
                blocks, stage_indices=[k], dim_in=dim_in
            )

    def _combiner_lookup(self, name: str, num_inputs: int) -> nn.Module:
        """Return combiner op corresponding to name

        Standard ops are stored in a lookup table. Alternatively, can use
        ops that have been registered.
        Inputs: (str) name
        Return: (nn.Module) op
        """
        # combiner_lookup = {
        #     "add": bb.TorchNLengthAdd,
        #     "choose_right": bb.ChooseRightPath,
        # }
        # if name in combiner_lookup:
        # if name == "add":
        #     return combiner_lookup[name]()

        if name == "add":
            return bb.TorchNLengthAdd(num_inputs)
        if name == "choose_right":
            return bb.ChooseRightPath()

        op = STAGE_COMBINERS_REGISTRY.get(name)
        if op is not None:
            return op()
        raise ValueError(f"Unrecognized combiner: {name}")

    def build_stage_combiners(
        self,
        stage_combiner_num_inputs: List[int],
        stage_combiners: Optional[List],
    ) -> None:
        """Initialize stage combiners

        Inputs: (list(str)) stage combiners
        Return: None
        """
        if stage_combiners is None:
            stage_combiners = ["add"] * self.num_resolutions

        # pyre-fixme[16]: `FBNetFPNBuilder` has no attribute `stage_combiners`.
        self.stage_combiners = nn.ModuleList(
            self._combiner_lookup(x, num_inputs)
            for x, num_inputs in zip(stage_combiners, stage_combiner_num_inputs)
        )


class FBNetFPN(nn.Module):
    def __init__(
        self,
        stages: List,
        stage_combiners: List,
        output_channels: List,
        combiner_path: Optional[str],
    ):
        """
        Args:
            stages: modules that will be called in the forward pass as shown
            in the hourglass network graph, should be given in a nn.ModuleList
            of nn.ModuleLists
                [
                 [stage0, stage1, stage2, stage3, stage4],
                 [stage0, stage1, stage2, stage3, stage4],
                 ...
                ]

            stage_combiners: list of combine ops (e.g., ["add", "add", ...]) in
            order of cross-resolution to same-resolution
                [combiner_type(stage4, stage5, stage6), ...]

            combiner_path: ordering of arguments to combine, assuming that lower
            numbered stages have higher resolution features.
                stage_combiners = ["add", "add", ...]
                combiner_path = "high_res"
                => stage2_input = add(stage1, stage5)

                stage_combiners = ["add", "add", ...]
                combiner_path = "low_res":
                => stage2_input = add(stage5, stage1)
        """
        super().__init__()
        self.stages = copy.deepcopy(stages)
        self.stage_combiners = copy.deepcopy(stage_combiners)
        self.output_channels = output_channels
        self.num_resolutions = len(stage_combiners)
        self.num_stages_per_resolution = 5
        self.combiner_path = combiner_path

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass is generated programatically to be independent of the
        number of spatial resolutions. The only requirement is that there are
        5 stages per spatial resolution

        Since blocks depend on >=1 other blocks, we have to make sure of the
        ordering of block calculations.
        """
        # store results
        data = [
            [None] * self.num_stages_per_resolution for _ in range(self.num_resolutions)
        ]

        for j in range(self.num_resolutions):
            for i in range(self.num_stages_per_resolution):
                if i == 0 or i == 1:
                    data[j][i] = self.stages[j][i](x[j * 2 + i])
                elif i == 2:
                    a, b, c = data[j][0], data[j][1], data[j - 1][-1]
                    inputs = [t for t in [a, b, c] if t is not None]
                    combined_result = self.stage_combiners[j](inputs)
                    data[j][i] = self.stages[j][i](combined_result)
                elif i == 3 or i == 4:
                    if self.stages[j][i] is not None:
                        data[j][i] = self.stages[j][i](data[j][2])
                # if data[j][i] is not None:
                #     print("res", j, "stage", i, "shape", data[j][i].shape)
        output = [data[j][-2] for j in range(self.num_resolutions)]
        if self.combiner_path == "low_res":
            output = output[::-1]

        # pyre-fixme[7]: Expected `List[torch.Tensor]` but got `List[None]`.
        return output
