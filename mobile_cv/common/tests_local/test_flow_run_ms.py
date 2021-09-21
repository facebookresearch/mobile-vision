#!/usr/bin/env python3

import sys

# sys.path.append('mobile-vision/common/tools/')
sys.path.append("mobile-vision")
import mobile_cv.common.fb.flu.fblearner_launch_utils as flu


flu.set_use_canary_flow()
# flu.set_debug_local()
flu.set_secure_group("team_ai_mobile_cv")


def test_build_flow():
    flow_bulder = flu.WorkflowOpBuilder(
        "test_build_flow",
        workflow_name="mobile_vision.detectron2go.core",
        workflow_input={},
        workflow_schedule_kwargs={
            "run_as_secure_group": "fblearner_flow_integration_tests"
        },
    )
    version = flow_bulder.build_flow()  # noqa


def test_flow_run_ms_d2go():
    workflow_name = "mobile_vision.detectron2go.core.workflow.e2e_workflow@mobile_vision.detectron2go.core"
    workflow_input = {
        "config_file": "detectron2go://faster_rcnn_fbnetv3a_dsmask_C4.yaml",
        "e2e_train": {
            "dist_config": {
                "num_machines": 1,
                "num_processes_per_machine": 8,
            },
            "resources": {"memory": "200g", "capabilities": ["GPU_V100_HOST"]},
        },
    }
    workflow_schedule_kwargs = {
        "run_as_secure_group": "oncall_mobile_cv",
        "entitlement": "bigbasin_atn_arvr",
        "oncall": "mobile_cv",
    }
    secure_group_name = "oncall_mobile_cv"
    flu.flow_run_ms(
        name="test_flow_run_ms",
        workflow_name=workflow_name,
        workflow_input=workflow_input,
        workflow_schedule_kwargs=workflow_schedule_kwargs,
        secure_group_name=secure_group_name,
        build_flow=True,
    )


def test_flow_run_ms_hello_world():
    workflow_name = "tests.canary.helloworld.HelloWorld@fblearner.flow.tests"
    workflow_input = {}
    workflow_schedule_kwargs = {
        "run_as_secure_group": "oncall_mobile_cv",
        "entitlement": "default",
        "oncall": "mobile_cv",
    }
    secure_group_name = "oncall_mobile_cv"
    flu.flow_run_ms(
        name="test_flow_run_ms",
        workflow_name=workflow_name,
        workflow_input=workflow_input,
        workflow_schedule_kwargs=workflow_schedule_kwargs,
        secure_group_name=secure_group_name,
        build_flow=True,
    )


if __name__ == "__main__":
    # test_build_flow()
    # test_flow_run_ms_d2go()
    test_flow_run_ms_hello_world()
