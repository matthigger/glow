import hglm

exp = hglm.experiment.Experiment.from_gauss(seed=0,
                                            shape=(5, 5, 5),
                                            a=2,
                                            b=2,
                                            num_img=100)

hglm.experiment.AnalysisHGLM(exp,
                             n_perm=20,
                             n_perm_adj=10,
                             n_perm_tailor=20,
                             alpha_fwer=.05,
                             alpha_tailor=.05,
                             min_size=1)
