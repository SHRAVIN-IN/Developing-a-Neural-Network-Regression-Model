# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: JANA SHRAVIN S

### Register Number: 212224243003

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:\\Users\\S.JANA SHRAVIN\\Downloads\\house_price_regression_dataset.csv")
print(df)
x=df[["Area_sqft","Bedrooms","Bathrooms","House_Age_Years"]].values
y=df[["Price_Lakhs"]].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.33,random_state=33)
scaler1=StandardScaler()
xt=scaler1.fit_transform(xt)
xst=scaler1.fit_transform(xst)
xt1=torch.FloatTensor(xt)
xst1=torch.FloatTensor(xst)
yt1=torch.FloatTensor(yt)
yst1=torch.FloatTensor(yst)
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 70),   # 4 input features → 70 hidden units
            nn.ReLU(),
            nn.Linear(70, 40),
            nn.ReLU(),
            nn.Linear(40, 10),   # output: single price value
            nn.ReLU(),
            nn.Linear(10, 1)     # output: single price value
        )

    def forward(self, x):
        return self.network(x)
model=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.000001)
epochs=10000
for i in range(epochs):
    optimizer.zero_grad()
    pred=model(xt1)
    loss=criterion(pred,yt1)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(f"Epochs:{i} Loss:",loss.item())
with torch.no_grad():
    pred=model(xst1)
    loss=criterion(pred,yst1)
    print(loss.item())
data = scaler1.transform([[1100, 2, 2, 12]])
input_tensor = torch.tensor(data, dtype=torch.float32)
# Ensure the model matches the 4 input features used by scaler1
model.eval()
with torch.no_grad():
    pred = model(input_tensor)
print(pred.item())


```

### Dataset Information
Include screenshot of the generated data

### OUTPUT
<img width="1592" height="458" alt="image" src="https://github.com/user-attachments/assets/d1b10398-9c04-4ee8-9f6f-f0a957700e28" />

### Training Loss Vs Iteration Plot
<img width="1067" height="872" alt="image" src="https://github.com/user-attachments/assets/11ccd573-a10f-4dd5-83ff-00299c9ae582" />
<img width="1003" height="857" alt="image" src="https://github.com/user-attachments/assets/4c1ce46a-9fb8-44b0-a366-57b2f9bde7f9" />


### New Sample Data Prediction
****<img width="1592" height="458" alt="image" src="https://github.com/user-attachments/assets/3030190a-ad5e-4d9c-bbf8-f71d2a1448b9" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
