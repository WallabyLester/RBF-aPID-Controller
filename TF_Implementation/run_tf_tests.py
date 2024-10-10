import unittest

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.discover(start_dir='test', pattern='*.py'))

runner = unittest.TextTestRunner()
runner.run(suite)