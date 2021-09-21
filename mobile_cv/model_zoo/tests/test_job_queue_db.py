#!/usr/bin/env python3

import unittest

import mobile_cv.model_zoo.job_queue.db as jd
from mobile_cv.common.misc.file_utils import make_temp_directory


class TestModelZooJobQueueDB(unittest.TestCase):
    def test_insert_job(self):
        """Check that db is created"""
        with make_temp_directory("test") as db_dir:
            job_db = jd.JobDB(db_dir)
            job_db.add(jd.JobItem("project_name", "model_info0", ["model_tags0"]))
            items = job_db.query("wait")
            self.assertEqual(len(items), 1)
            self.assertIsNotNone(items[0].ts)

            # list of items
            job_db.add([jd.JobItem("project_name", "model_info1", ["model_tags1"])])
            items = job_db.query("wait")
            self.assertEqual(len(items), 2)

    def test_insert_dict(self):
        """Check modelinfo can be dict"""
        gt_modelinfo = {"model_info": 0}
        with make_temp_directory("test") as db_dir:
            job_db = jd.JobDB(db_dir)
            job_db.add(jd.JobItem("project_name", gt_modelinfo, ["tag0", "tag1"]))
            items = job_db.query("wait")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].model_info, gt_modelinfo)

    def test_update_job_state(self):
        """Check job state can be updated"""
        with make_temp_directory("test") as db_dir:
            job_db = jd.JobDB(db_dir)
            job_db.add(jd.JobItem("project_name", "modelinfo", ["model_tags0"]))
            items = job_db.query("wait")
            self.assertEqual(len(items), 1)
            job_id = items[0].id
            job_db.update_job_state(job_id, "run")
            items = job_db.query("run")
            self.assertEqual(len(items), 1)
