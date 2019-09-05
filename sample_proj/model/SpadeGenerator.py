

# 2入力1出力
class Spade(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        self.batch_norm = nn.SyncBatchNorm()
        self.preprocess = nn.Sequential(
            nn.Upsample(size=(h,w)),
            nn.Conv2d(),
            nn.ReLU()
        )
        self.gamma_conv = nn.Conv2d()
        self.beta_conv = nn.Conv2d()

    def forward(self,x,seg):
        x2 = self.batch_norm(x2)
        seg2 = self.preprocess(seg)
        beta, gamma = self.beta_conv(seg2)+1, self.gamma_conv(seg2)
        y = x2*beta+gamma
        return y


class SpadeResBlock(nn.Module):

    def _make_module(self):
        return nn.Sequential(
            nn.Conv2d(),
            nn.ReLU()
        )

    def __init__(self):
        super().__init__()
        self.spadeA1 = Spade()
        self.convA1 = self._make_module()
        self.spadeA2 = Spade()
        self.convA2 = self._make_module()
        self.spadeB = Spade()
        self.convB = self._make_module()

    def forward(self,x,seg):
        A1_1 = self.spadeA1(x,seg)
        A1_2 = self.convA1(A1_1)
        A2_1 = self.spadeA2(A1_2,seg)
        A2_2 = self.convA2(A2_1)

        B_1 = self.spadeB(x,seg)
        B_2 = self.convB(B_1)

        y = A2_2 + B
        return y


class SpadeGenerator(nn.Module):

    def __init__(self,size=1024):
        self.up = nn.Upsample(scale_factor=2)
        self.decoder = nn.ModuleList(
            [SpadeResBlock(h=2**(i+2),w=2**(i+2)) for i in range(5)])

        self.toImage = nn.Sequential(nn.LeakyReLU(),
                                        nn.Conv2d(),
                                        nn.Tanh())
        self.step = 1

    def forward(self,z,seg):
        '''
        z : random or None
        '''
        z = F.interpolate(seg,size=(self.startH,self.startW)) if z is None else z

        decoded = [ (l.append( self.up(f(l[-1])) ),l[-1])[-1]
                        for l in [[z]] for f in self.decoder[:self.step]]
        
        x = self.toImage(decoded[-1])
        return x