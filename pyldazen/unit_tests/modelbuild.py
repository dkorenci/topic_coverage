from pyldazen.unit_tests.data import *
from pyldazen.build.LdaGibbsBuilder import LdaGibbsBuilder
from pyldazen.build.LdaGibbsBuildData import LdaGibbsBuildData

import unittest
from unittest import TestCase

class BuildTest(TestCase):
    ''''''

    def runBuild(self, corpus, buildData):
        buildData.documents = corpus
        LdaGibbsBuilder()(buildData)

    # todo: put realistinc corpus and unrealistic corpus in different suites
    def testBuildRealistic(self):
        self.runBuild(croelectNews(1000), LdaGibbsBuildData(50, None, None, 1.0, 0.01))
        self.assertTrue(True)

    def testBuildSingleton(self):
        self.runBuild(singleton(), LdaGibbsBuildData(50, None, None, 1.0, 0.01))
        self.assertTrue(True)

    def testBuildManySingleton(self):
        self.runBuild(manySingletons(), LdaGibbsBuildData(50, None, None, 1.0, 0.01))
        self.assertTrue(True)

    def testBuildSingleton(self):
        self.runBuild(singleton(), LdaGibbsBuildData(50, None, None, 1.0, 0.01))
        self.assertTrue(True)

    def testCorpus1(self):
        self.runBuild(testCorpus1(), LdaGibbsBuildData(50, None, None, 1.0, 0.01))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

