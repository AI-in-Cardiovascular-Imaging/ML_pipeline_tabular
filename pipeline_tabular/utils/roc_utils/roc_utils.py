import warnings
import numpy as np


def compute_roc_aucopt(fpr, tpr, thr, costs, X=None, y=None, auto_flip=False):
    """
    Given the false positive rates fpr(thr) and true positive rates tpr(thr)
    evaluated for different thresholds thr, the AUC is computed by simple
    integration.

    Besides AUC, the optimal threshold is computed that maximizes some cost
    criteria. Argument costs is expected to be a dictionary with the cost
    type as key and a functor f(fpr, tpr) as value.

    If X and y are provided (optional), the resulting prediction
    accuracy is also computed for the optimal point.

    The function returns a struct as described in compute_roc().
    """
    # It's possible that the last element [-1] occurs more than once sometimes.
    thr_no_last = np.delete(thr, np.argwhere(thr == thr[-1]))
    if len(np.unique(thr_no_last)) != len(thr_no_last):
        warnings.warn("thr should contain only unique values.")

    # Assignment cost ratio, see reference below and get_objective().
    m = 1

    auc = np.trapz(x=fpr, y=tpr)
    flipped = False
    if auto_flip:
        if auc < 0.5:
            auc, flipped = 1 - auc, True  # Mark as flipped.
            fpr, tpr = tpr, fpr  # Flip the data!

    opd = {}
    for cost_id, cost in costs.items():
        # NOTE: The following evaluation is optimistic if x contains duplicated
        #       values! This is best seen in an example. Let's try to optimize
        #       the prediction accuracy acc=(TP+TN)/(T+P). Let x and y be
        #           x = 3 3 3 4 5 6
        #           y = F T T T T T.
        #       The optimal threshold is 3. It binarizes the data as follows,
        #           b = F F F T T T     = x > 3
        #           y = F T T T T T     => acc = 4/6
        #       However, the brute-force optimization permits to find a split
        #       in-between the duplicated values of x.
        #           o = F T T T T T
        #           y = F T T T T T     => acc = 1.0
        # NOTE: The effect of this optimism is negligible if the ordering of
        #       the outcome y for duplicated values in x is randomized. This
        #       is typically the case for natural data.
        #       If the "parametrization" thr does not contain equal points,
        #       the problem is not apparent.
        costs = list(map(cost, fpr, tpr))
        ind = np.argmax(costs)
        opt = thr[ind]
        if X is not None and y is not None:
            # Prediction accuracy (if X available).
            opa = sum((X > opt) == y) / float(len(X))
            opa = (1 - opa) if flipped else opa
        else:
            opa = None
        # Now that we got the index, flip back to extract the data.
        if flipped:
            fpr, tpr = tpr, fpr
        # Characterize optimal point.
        opo = costs[ind]
        opp = (fpr[ind], tpr[ind])
        q = tpr[ind] - m * fpr[ind]
        opq = ((0, 1), (q, m + q))

        opd[cost_id] = StructContainer(
            ind=ind,  # index of optimal point
            opt=opt,  # optimal threshold
            opp=opp,  # optimal point (fpr*, tpr*)
            opa=opa,  # prediction accuracy
            opo=opo,  # objective value
            opq=opq,
        )  # line through opt point

    struct = StructContainer(fpr=fpr, tpr=tpr, thr=thr, auc=auc, opd=opd, inv=flipped)
    return struct


class StructContainer:
    """
    Build a type that behaves similar to a struct.

    Usage:
        # Construction from named arguments.
        settings = StructContainer(option1 = False,
                                   option2 = True)
        # Construction from dictionary.
        settings = StructContainer({"option1": False,
                                    "option2": True})
        print(settings.option1)
        settings.option2 = False
        for k,v in settings.items():
            print(k,v)
    """

    def __init__(self, dictionary=None, **kwargs):
        if dictionary is not None:
            assert isinstance(dictionary, (dict, StructContainer))
            self.__dict__.update(dictionary)
        self.__dict__.update(kwargs)

    def __iter__(self):
        for i in self.__dict__:
            yield i

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return sum(1 for k in self.keys())

    def __repr__(self):
        return "struct(**%s)" % str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def items(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield (k, v)

    def keys(self):
        for k in self.__dict__:
            if not k.startswith("_"):
                yield k

    def values(self):
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield v

    def update(self, data):
        self.__dict__.update(data)

    def asdict(self):
        return dict(self.items())

    def first(self):
        # Assumption: __dict__ is ordered (python>=3.6).
        key, value = next(self.items())
        return key, value

    def last(self):
        # Assumption: __dict__ is ordered (python>=3.6).
        # See also: https://stackoverflow.com/questions/58413076
        key = list(self.keys())[-1]
        return key, self[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def setdefault(self, key, default=None):
        return self.__dict__.setdefault(key, default)
