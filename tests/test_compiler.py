from simwave import Compiler
import pytest


class TestCompiler:

    def test_cc(self):
        for cc in ['gcc', 'icc', 'clang', 'cc']:
            compiler = Compiler(cc=cc)
            assert compiler.cc == cc

    @pytest.mark.parametrize(
        'cc, language, cflags, expected', [
            ('gcc', 'c', None, '-O3 -fPIC -Wall -std=c99 -shared'),
            ('gcc', 'c', '-O3 -fPIC', '-O3 -fPIC -shared'),
            ('icc', 'c', '-O3 -shared', '-O3 -shared'),
            ('icc', 'c', '-O3', '-O3 -shared'),
            ('clang', 'c', None, '-O3 -fPIC -Wall -std=c99 -shared'),
            ('gcc', 'cpu_openmp', '-O3 -fPIC', '-O3 -fPIC -shared -fopenmp'),
            ('pgcc', 'cpu_openmp', '-shared', '-shared -mp'),
            ('icc', 'cpu_openmp', '-O3', '-O3 -shared -openmp'),
            ('clang', 'cpu_openmp', '-O3', '-O3 -shared -fopenmp'),
            ('foo', 'cpu_openmp', '-O3', '-O3 -shared')
        ]
    )
    def test_cflags(self, cc, language, cflags, expected):
        compiler = Compiler(cc=cc, language=language, cflags=cflags)
        assert compiler.cflags == expected

    def test_language(self):
        for language in ['c', 'cpu_openmp']:
            compiler = Compiler(language=language)
            assert compiler.language == language

    def test_language_fail(self):
        with pytest.raises(ValueError):
            Compiler(language='cpu_mpi')

    @pytest.mark.parametrize(
        'cc, omp_flag', [
            ('gcc', '-fopenmp'), ('icc', '-openmp'), ('pgcc', '-mp'),
            ('clang', '-fopenmp'), ('foo', None)
        ]
    )
    def test_get_openmp_flag(self, cc, omp_flag):

        compiler = Compiler(cc=cc)
        assert compiler.get_openmp_flag() == omp_flag
