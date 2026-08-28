"""Shared interfaces between harness, controller, executor, and methods.

This module is the single source of truth for the data structures and
protocols that cross package boundaries (e.g. the shape of a proposed
method, a run result, a gate verdict, a journal entry). All four packages
should import from here rather than depending on each other's internals.

Placeholder for now — no contracts defined yet.
"""
