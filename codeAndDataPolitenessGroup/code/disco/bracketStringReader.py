from discodop import treebank

class BracketStringReader(treebank.BracketCorpusReader):
    def __init__(self, treeStrings, encoding='utf-8', ensureroot=None, punct=None,
            headrules=None, headfinal=True, headreverse=False, markheads=False,
            removeempty=False, functions=None, morphology=None, lemmas=None):
        self.removeempty = removeempty
        self.ensureroot = ensureroot
        self.reverse = headreverse
        self.headfinal = headfinal
        self.markheads = markheads
        self.functions = functions
        self.punct = punct
        self.morphology = morphology
        self.lemmas = lemmas
        self.headrules = readheadrules(headrules) if headrules else {}
        self._encoding = encoding
        self._filenames = []
        for opts, opt in (
                ((None, 'leave', 'add', 'replace', 'remove', 'between'),
                    functions),
                ((None, 'no', 'add', 'replace', 'between'), morphology),
                ((None, 'no', 'move', 'moveall', 'remove', 'root'), punct),
                ((None, 'no', 'add', 'replace', 'between'), lemmas)):
            if opt not in opts:
                raise ValueError('Expected one of %r. Got: %r' % (opts, opt))
        self._block_cache = None
        self._trees_cache = None
        self.treeStrings = treeStrings


    def _read_blocks(self):
        for n, block in enumerate(self.treeStrings, 1):
            yield n, block
