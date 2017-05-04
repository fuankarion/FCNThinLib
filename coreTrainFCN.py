def trainNetworkLog(solver, niter, batchesForTraining, targetLogFile, test_iters, logsPerEpoch, smallSetProportion): 
    
    train_loss = 0.0
    train_accuracy = 0.0
    smallSetIters = int(round(test_iters / smallSetProportion))#test with a small set inbetween epochs
    
    for it in range(niter):
        solver.step(1)
        print('Iteration ', it)
        
        train_loss = train_loss + solver.net.blobs['loss'].data   
        train_accuracy = train_accuracy + solver.net.blobs['accuracy'].data   

        #get loss and accuracy, by doing test_iters forward pass and avergaing results per batch
        if (it % round(batchesForTraining / logsPerEpoch)) == 0 and it > 0:# logsPerEpoch
            if (it % (round(batchesForTraining / logsPerEpoch) * logsPerEpoch))  != 0:#Do small test?
                adaptedTest_iters = smallSetIters
            else:
                adaptedTest_iters = test_iters
                                
            test_acc = 0.0
            test_loss = 0.0
            for i in range(adaptedTest_iters):
                solver.test_nets[0].forward()#TODO what if we use more tha 1 test net
                accuracyTemp = solver.test_nets[0].blobs['accuracy'].data
                test_acc = test_acc + accuracyTemp
                
                lossTemp = solver.test_nets[0].blobs['loss'].data
                test_loss = test_loss + lossTemp
               
                print('On test stage iter : ', i, ' accuracy ', accuracyTemp, ' loss ', lossTemp)
            test_acc = test_acc / adaptedTest_iters
            test_loss = test_loss / adaptedTest_iters
          
            train_accuracy = train_accuracy / (batchesForTraining / logsPerEpoch)
            train_loss = train_loss / (batchesForTraining / logsPerEpoch)
                      
            print 'iter ', it, 'train loss:', train_loss, 'train accuracy ', train_accuracy, 'test losss', test_loss, 'test accuracy:', test_acc
            print ''
            
            
            if (it % (round(batchesForTraining / logsPerEpoch) * logsPerEpoch))  != 0:#write small test
                with open(targetLogFile, 'a') as myfile:
                    myfile.write(str(it) + ',' + str(train_loss) + ',' + str(train_accuracy) + ',' + str(test_loss) + ',' + str(test_acc) + '\n')
            else:
                with open(targetLogFile, 'a') as myfile:
                    myfile.write(str(it) + ',' + str(train_loss) + ',' + str(train_accuracy) + ',' + str(test_loss) + ',' + str(test_acc) + ',X \n')
           
                
            train_loss = 0.0
            train_accuracy = 0.0
               
    print 'You Actually got here :)'
    