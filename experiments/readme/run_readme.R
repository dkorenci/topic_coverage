library(ReadMe)

# set base foder and labels
baseFolder = '/datafast/pymedialab/readme/'
labels = c('civil rights movement',
             'lgbt rights',  'police brutality',
                'chapel hill', 'reproductive rights',
                'violence against women', 'death penalty', 'surveillance',
                'gun rights', 'net neutrality', 'marijuana', 'vaccination')

# run readme for each label (folder)
# resFile <- paste(baseFolder,'output.txt',sep='')
# file.remove(resFile)
# for(l in labels) {    
#   inputFolder <- paste(baseFolder,l,sep='')
#   setwd(inputFolder)
#   undergrad.results <- undergrad()
#   undergrad.preprocess <- preprocess(undergrad.results)
#   readme.results <- readme(undergrad.preprocess)    
#   #res <- runReadme(inputFolder)
#   print(l)
#   print(res)
#   cat(paste(l,' : ',''), file=resFile, append = T)
#   cat(res, file=resFile, append = T)
#   cat('\n' , file=resFile, append = T)
# }

inputFolder <- paste(baseFolder,'civil rights movement',sep='')
runReadme <- function(inputFolder) {
  setwd(inputFolder)  
  undergrad.results <- undergrad(printit=F)
  undergrad.preprocess <- preprocess(undergrad.results)
  capture.output(readme.results <- readme(undergrad.preprocess, printit=FALSE))
#  capture.output(undergrad.results <- undergrad(), file='/dev/null')
#  capture.output(undergrad.preprocess <- preprocess(undergrad.results), file='/dev/null')
#  capture.output(readme.results <- readme(undergrad.preprocess), file='/dev/null')
  return(readme.results$est.CSMF)
}

#sol <- runReadme(inputFolder)

runReadmeAvg <- function(inputFolder, rndseed=5678, runs=5) {
  setwd(inputFolder)  
  undergrad.results <- undergrad(printit=F)
  undergrad.preprocess <- preprocess(undergrad.results)
  avgProp <- 0
  for (s in rndseed:(rndseed+runs-1)) {
    set.seed(s)
    capture.output(readme.results <- readme(undergrad.preprocess, printit=FALSE))
    prop <- readme.results$est.CSMF[2]
    avgProp <- avgProp + prop    
  }
  prop <- avgProp / runs  
  return(prop)
}

#sol <- runReadmeAvg(inputFolder, rndseed=1233, runs=4)
