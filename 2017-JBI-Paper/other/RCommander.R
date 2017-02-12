library(survival, pos=4)
original <- read.csv("data/manually_binarized_immuno.csv")
preprocessed <- read.csv("data/automatically_binarized_immuno.csv")
autoencoded <- read.csv("data/autoencoded_binarized_immuno.csv")
output<-"Recurrence&&&&&&Mortality&&&&&"
output<-list(output, "Original&&&Preprocessed&&Autoencoded&Original&&&Preprocessed&&Autoencoded")
output<-list(output, "Marcador&CoxPValue&Prop. Haz. Assumption&CoxPValue&Prop. Haz. Assumption&CoxPValue&Prop. Haz. Assumption&CoxPValue&Prop. Haz. Assumption&CoxPValue&Prop. Haz. Assumption&CoxPValue&Prop. Haz. Assumption")
for (i in 1:21) {
	time.dep <- coxph( Surv(TREC, Recidiva)~original[,i],
            original, method="breslow", na.action=na.exclude)

	time.dep.zph <- cox.zph(time.dep, transform = 'log')

	ori<-paste(summary(time.dep)[7]$coefficients[5], "&", time.dep.zph$table[3])

	time.dep <- coxph( Surv(TREC, Recidiva)~preprocessed[,i],
            preprocessed, method="breslow", na.action=na.exclude)

	time.dep.zph <- cox.zph(time.dep, transform = 'log')

	prep<-paste(summary(time.dep)[7]$coefficients[5], "&", time.dep.zph$table[3])

	time.dep <- coxph( Surv(TREC, Recidiva)~autoencoded[,i],
            autoencoded, method="breslow", na.action=na.exclude)

	time.dep.zph <- cox.zph(time.dep, transform = 'log')

	auto<-paste(summary(time.dep)[7]$coefficients[5], "&", time.dep.zph$table[3])

	time.dep <- coxph( Surv(TMORT, Mortalidad)~original[,i],
            original, method="breslow", na.action=na.exclude)

	time.dep.zph <- cox.zph(time.dep, transform = 'log')

	ori2<-paste(summary(time.dep)[7]$coefficients[5], "&", time.dep.zph$table[3])

	time.dep <- coxph( Surv(TMORT, Mortalidad)~preprocessed[,i],
            preprocessed, method="breslow", na.action=na.exclude)

	time.dep.zph <- cox.zph(time.dep, transform = 'log')

	prep2<-paste(summary(time.dep)[7]$coefficients[5], "&", time.dep.zph$table[3])

	time.dep <- coxph( Surv(TMORT, Mortalidad)~autoencoded[,i],
            autoencoded, method="breslow", na.action=na.exclude)

	time.dep.zph <- cox.zph(time.dep, transform = 'log')

	auto2<-paste(summary(time.dep)[7]$coefficients[5], "&", time.dep.zph$table[3])

	output<-list(output, paste(colnames(autoencoded)[i], "&", ori, "&", prep, "&", auto, "&", ori2, "&", prep2, "&", auto2))
}
write.csv(unlist(output), 'data/results.csv')



