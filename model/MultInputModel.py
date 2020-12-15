from model.LSTMoutput import getLSTMModel, LSTMoutput
from training.FCFNtrain import trainFCFN, evaluateFCFN

def MultiInputModel(item_train_dl,item_valid_dl,item_vocab,user_train_dl,user_valid_dl,user_vocab,embedding_dim,hidden_dim):

  item_LSTMmodel, user_LSTMmodel = getLSTMModel(item_train_dl, item_valid_dl,item_vocab,user_train_dl, user_valid_dl,user_vocab,100, 70)

  FCFN_input = LSTMoutput(item_LSTMmodel, user_LSTMmodel, item_train_dl, user_train_dl)

  epochs = 5
  lr = 1e-4
  indim = FCFN_input.shape[1]
  outdim = 5
  drate = 0.7
  batch_size = 16

  #abstract out hard-coding of 432
  X_tensor = FCFN_input
  Y_tensor = torch.from_numpy(np.array(y[:432]))


  dataset = TensorDataset(X_tensor,Y_tensor)
  train_size = int(0.8*len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size,val_size])

  #create training loader and validation loader
  train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
  val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

  print("Feeding flattened output into FCFN....")
  net = FCFN(indim,outdim,drate)
  optimizer = torch.optim.SGD(net.parameters(),lr=lr)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(epochs):
    train_loss, train_acc = trainFCFN(net, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluateFCFN(net, val_loader, criterion)
    print(f"\n\nEpoch # {epoch+1}")
    print(f"Training Set:\nAverage Loss: {train_loss:0.4f}  |  Average Accuracy: {train_acc:0.4f}")
    print(f"\nValidation Set:\nAverage Loss: {valid_loss:0.4f}  |  Average Accuracy: {valid_acc:0.4f}")
