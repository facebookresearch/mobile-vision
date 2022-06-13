#!/usr/bin/env python3
import unittest

from mobile_cv.common.misc.mixin import dynamic_mixin, remove_dynamic_mixin


class Person:
    def __init__(self, age: int = 1, name: str = "MyName"):
        self.age = 1
        self.name = name

    def get_name(self):
        return self.name


class Student:
    def dynamic_mixin_init(self, school="MySchool"):
        self.school = school

    def remove_dynamic_mixin(self):
        del self.school

    def get_name(self):
        return f"student_{self.name}"


class TestDynamicMixin(unittest.TestCase):
    def test_dynamic_mixin(self):
        """Check new class is applied to object"""
        person = Person()
        dynamic_mixin(person, Student, init_new_class=False)
        self.assertEqual(person.__class__.__name__, "Person_Student")
        self.assertTrue(Student in person.__class__.mro())
        self.assertTrue(hasattr(person, "_original_model_class"))

    def test_init(self):
        """Check init"""
        person = Person()
        dynamic_mixin(person, Student, init_new_class=True)
        self.assertTrue(hasattr(person, "school"))
        self.assertEqual(person.school, "MySchool")

        person = Person()
        dynamic_mixin(
            person, Student, init_new_class=True, init_dict={"school": "NewSchool"}
        )
        self.assertEqual(person.school, "NewSchool")

    def test_remove(self):
        """Check removing dyanmic mixed in class"""
        person = Person()
        dynamic_mixin(person, Student, init_new_class=True)
        remove_dynamic_mixin(person)
        self.assertFalse(hasattr(person, "_original_model_class"))
        self.assertFalse(hasattr(person, "school"))

    def test_override(self):
        """Check that mixed in class will override methods"""
        person = Person()
        self.assertEqual(person.get_name(), "MyName")
        dynamic_mixin(person, Student, init_new_class=False)
        self.assertEqual(person.get_name(), "student_MyName")
