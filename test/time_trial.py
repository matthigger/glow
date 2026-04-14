import glow

exp = glow.experiment.Experiment.from_gauss(seed=0,
                                            shape=(100, 100),
                                            a=2,
                                            b=2,
                                            num_img=100)

glow.analysis.AnalysisGLOW(exp,
                             n_perm_fwer=20,
                             alpha_fwer=.05,
                             min_size=1)
