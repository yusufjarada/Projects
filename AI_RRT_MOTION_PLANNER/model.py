import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def extract_data(file_path):
    data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        start = None
        goal = None
        waypoints = []
        for line in lines:
            line = line.strip()
            if line.startswith("Start Configuration:"):
                start = list(map(float, line.split(": (")[1][:-1].split(", ")))
            elif line.startswith("Goal Configuration:"):
                goal = list(map(float, line.split(": (")[1][:-1].split(", ")))
            elif line.startswith("Waypoint:"):
                waypoint = list(map(float, line.split(": ")[1].split(" - ")[0][1:-1].split(", ")))
                cost = float(line.split(": ")[2])
                waypoints.append((waypoint, cost))
            elif line == "":
                if start is not None and goal is not None and waypoints:
                    for waypoint, cost in waypoints:
                        data.append((start, goal, waypoint, cost))
                    start = None
                    goal = None
                    waypoints = []

    features = []
    labels = []
    for datapoint in data:
        start = np.array(datapoint[0])
        goal = np.array(datapoint[1])
        waypoint = np.array(datapoint[2])
        x = (np.concatenate((start, goal, waypoint), axis=0))
        y = (np.array(datapoint[3]))
        features.append(np.ndarray.tolist(x))
        labels.append(np.ndarray.tolist(y))

    return features, labels

def save_model(model, optimizer, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        # Add more keys if needed, such as model architecture configuration
    }
    torch.save(checkpoint, filepath)

# Load model function
def load_model(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load more keys if needed
    model.eval()  # Set model to evaluation mode
    return model

# Define your neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_prob=0.5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# Example usage:

if __name__ == "__main__":
    features, labels = extract_data("data/env1")

    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"

    device = torch.device(dev)
    # Convert your data to PyTorch tensors
    X_tensor = torch.tensor(features, dtype=torch.float32)  # Features (start and goal configurations)
    y_tensor = torch.tensor(labels, dtype=torch.float32)  # Labels (waypoint costs)

    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=23)


    # Define model parameters
    input_size = 9  # Define based on the number of features
    hidden_size1 = 64
    hidden_size2 = 32
    hidden_size3 = 16
    output_size = 1

    # Create an instance of the model
    model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # mean squared error
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train the model
    num_epochs = 2500

    import matplotlib.pyplot as plt

    # Train the model
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        outputs = outputs.view(-1)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store training loss
        train_losses.append(loss.item())

        # Compute test loss
        with torch.no_grad():
            y_pred_test = model(X_test)
            y_pred_test = y_pred_test.view(-1)
            test_loss = criterion(y_pred_test, y_test)
            test_losses.append(test_loss.item())

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    # Plotting the training and test losses
    # Plotting the training loss
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    # Plotting the test loss
    plt.plot(range(num_epochs), test_losses, label='Test Loss', color='orange')  # You can customize the color if needed
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = y_pred.view(-1)  # This fixed a sizing error, im not sure why
        test_loss = criterion(y_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
        # Calculate R^2
        r2 = r2_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        print(f'Coefficient of Determination (R^2): {r2:.4f}')


    save_model(model, optimizer, 'model_checkpoint.pth')
