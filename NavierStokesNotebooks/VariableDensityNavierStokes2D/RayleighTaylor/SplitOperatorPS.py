from proteus.SplitOperator import Sequential_FixedStep_Simple, defaultSystem

class Sequential_FixedStep_SimplePS(Sequential_FixedStep_Simple):
    def __init__(self,modelList,system=defaultSystem,stepExact=True):
        Sequential_FixedStep_Simple.__init__(self,modelList,system,stepExact)
        self.modelList = modelList[:4]
