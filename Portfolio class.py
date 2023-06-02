import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import scipy.optimize as sco
import scipy.stats as scs
# import openpyxl as xl


class Portfolio:
    def __init__(self, rp, cov, weights, rm, std, mean, rf):
        self.rp = rp
        self.cov = cov
        self.weights = weights
        self.betta = self.Beta()
        self.rm = rm
        self.std = std
        self.mean = mean
        self.rf = rf
        self.Treynor = self.Treynor()
        self.JensensAlpha = self.JensensAlpha()
    def JensensAlpha(self):
        """
        Считает Альфу Йенса
        :return:
        """
        coeffJensensAlpha = self.rp - (self.rf + self.betta * (self.rm - self.rf))
        return coeffJensensAlpha

    def Treynor(self):
        """
        Считает коэффициент Тейнора
        """
        coeffTreynor = (self.rp - self.rf) / self.betta
        return coeffTreynor

    def Beta(self):
        """
        Находит Бетта коэфициент
        """
        coeffBeta = (np.cov(self.rp, self.rm)) / (np.std(self.rm))
        return coeffBeta

