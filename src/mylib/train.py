#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`mylib.train` contains classes:
- :class:`mylib.train.Trainer`

The :mod:`mylib.train` contains functions:
- :func:`mylib.train.cv_parameters`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class Trainer:
    r'''Base class for all trainer.'''
    def __init__(self, model, X_train, Y_train, X_val, Y_val):
        r'''Constructor method

        :param model: The class with fit and predict methods.
        :type model: object

        :param X_train: The array of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X_train: numpy.array
        :param Y_train: The array of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y_train: numpy.array
        :param X_val: The array of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X_val: numpy.array
        :param Y_val: The array of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y_val: numpy.array
        '''
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

    def train(self):
        r''' Train model
        '''
        self.model.fit(self.X_train, self.Y_train)

    def eval(self):
        r'''Evaluate model for initial validadtion dataset.
        '''
        return classification_report(
            self.Y_val, 
            self.model.predict(
                self.X_val))

    def test(self, X, Y):
        r"""Evaluate model for given dataset.
        
        :param X: The array of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: numpy.array
        :param Y: The array of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: numpy.array
        """
        return classification_report(
            X, self.model.predict(Y))


def cv_parameters(X_train, Y_train, X_val, Y_val):
    r'''Function for the experiment
    '''
    Cs = numpy.linspace(0.1, 200, 100)
    parameters = []
    accuracy = []
    for C in Cs:
        model = LogisticRegression(penalty='l1', solver='saga', C=1/C)
        model.fit(X_train, Y_train)

        accuracy.append(
            classification_report(
                X_val, 
                model.predict(
                    Y_val), 
                output_dict=True)['accuracy']
        )
        
        parameters.extend(model.coef_)

    return accuracy, parameters