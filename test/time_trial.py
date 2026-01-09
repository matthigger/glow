import glow

exp = glow.experiment.Experiment.from_gauss(seed=0,
                                            shape=(100, 100),
                                            a=2,
                                            b=2,
                                            num_img=100)

glow.experiment.AnalysisGLOW(exp,
                             n_perm=20,
                             n_perm_adj=10,
                             n_perm_prune=20,
                             alpha_fwer=.05,
                             alpha_prune=.05,
                             min_size=1)
