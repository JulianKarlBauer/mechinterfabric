#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest


def test_change_sign_of_columns():
    m = np.random.rand(3, 3)
    m_changed = np.einsum("j, ij->ij", np.array([1, -1, -1]), m)
    assert np.allclose(m_changed, np.array([m[:, 0], -m[:, 1], -m[:, 2]]).T)
