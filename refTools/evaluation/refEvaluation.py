from refTools.evaluation.tokenizer.ptbtokenizer import PTBTokenizer
from refTools.evaluation.bleu.bleu import Bleu
from refTools.evaluation.meteor.meteor import Meteor
from refTools.evaluation.rouge.rouge import Rouge
from refTools.evaluation.cider.cider import Cider

"""
Input: refer and Res = [{ref_id, sent}]

Things of interest
evalRefs  - list of ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']
eval      - dict of {metric: score}
refToEval - dict of {ref_id: ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']}
"""

class RefEvaluation:
    def __init__ (self, refer, Res):
        """
        :param refer: refer class of current dataset
        :param Res: [{'ref_id', 'sent'}]
        """
        self.evalRefs = []
        self.eval = {}
        self.refToEval = {}
        self.refer = refer
        self.Res = Res

    def evaluate(self):

        evalRefIds = [ann['ref_id'] for ann in self.Res]

        refToGts = {}
        for ref_id in evalRefIds:
            ref = self.refer.Refs[ref_id]
            gt_sents = [sent['sent'].encode('ascii', 'ignore').decode('ascii') for sent in ref['sentences']]  # up to 3 expressions
            refToGts[ref_id] = gt_sents
        refToRes = {ann['ref_id']: [ann['sent']] for ann in self.Res}

        print('tokenization...')
        tokenizer = PTBTokenizer()
        self.refToRes = tokenizer.tokenize(refToRes)
        self.refToGts = tokenizer.tokenize(refToGts)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.refToGts, self.refToRes)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setRefToEvalRefs(scs, self.refToGts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setRefToEvalRefs(scores, self.refToGts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalRefs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setRefToEvalRefs(self, scores, refIds, method):
        for refId, score in zip(refIds, scores):
            if not refId in self.refToEval:
                self.refToEval[refId] = {}
                self.refToEval[refId]["ref_id"] = refId
            self.refToEval[refId][method] = score

    def setEvalRefs(self):
        self.evalRefs = [eval for refId, eval in self.refToEval.items()]


if __name__ == '__main__':

    import os.path as osp
    import sys
    ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    sys.path.insert(0, osp.join(ROOT_DIR, 'lib', 'datasets'))
    from refer import REFER

    # load refer of dataset
    dataset = 'refcoco'
    refer = REFER(dataset, splitBy = 'google')

    # mimic some Res
    val_refIds = refer.getRefIds(split='test')
    ref_id = 49767
    print("GD: %s" % refer.Refs[ref_id]['sentences'])
    Res = [{'ref_id': ref_id, 'sent': 'left bottle'}]

    # evaluate some refer expressions
    refEval = RefEvaluation(refer, Res)
    refEval.evaluate()

    # print output evaluation scores
    for metric, score in refEval.eval.items():
        print('%s: %.3f'%(metric, score))

    # demo how to use evalImgs to retrieve low score result
    # evals = [eva for eva in refEval.evalRefs if eva['CIDEr']<30]
    # print 'ground truth sents'
    # refId = evals[0]['ref_id']
    # print 'refId: %s' % refId
    # print [sent['sent'] for sent in refer.Refs[refId]['sentences']]
    #
    # print 'generated sent (CIDEr score %0.1f)' % (evals[0]['CIDEr'])

    # print refEval.refToEval[8]















