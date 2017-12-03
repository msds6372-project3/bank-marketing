library(pheatmap)
bank <- read.csv('/Users/Jostein/Grad School/SMU/6372/project3/bank-marketing/data/new_bank.csv', header=TRUE)
#str(bank)

#heatmap <- pheatmap(bank)
#bank.clust <- cbind(bank, cluster = cutree(heatmap$tree_row, k = 10))


bank.mat <- as.matrix(t(bank))
colnames(bank.mat) <- names(bank)
myannot <- data.frame(bank)
myannot$y_yes <- as.factor(myannot$y_yes)
rownames(myannot) <- bank$x1

pheatmap(bank.mat, scale ="row", annotation_col = myannot, color=colorRampPalette(c('dark red','white','dark blue'))(100))

# Source: Professor Turner's 2DS Wall Post
#Formatting data for cleaner looking heatmap.
#Forcing the data to remain balanced to keep the colors on the map balanced
full.mat<-as.matrix(t(dat[,-(1:5)]))
#log transforming data, not required but sometimes helpful in gene expression 
lfull.mat<-log2(full.mat)
lfull.mat[lfull.mat>7]<-7

#Creating a data frame to add column labels to the heatmap
colnames(lfull.mat)<-dat$x1
myannot<-data.frame(dat[,c("Set","Censor")])
myannot$Censor<-as.factor(myannot$Censor)
rownames(myannot)<-dat$x1

pheatmap(lfull.mat,scale="row",annotation_col=myannot,color=colorRampPalette(c('dark red','white','dark blue'))(100))

iris <- data(iris)