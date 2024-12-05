import numpy as np
import Load_game_records



def extract_features(observation):
    grid = observation
    '''
    print(len(grid))
    print(len(grid[0]))
    print(grid[0][0])
    '''
    def cell_to_feature(cell):
        object_mapping = {
            'floor': 0,
            'wall': 1,
            'avatar': 2,
            'alien': 3,
            'bomb': 4,
            'portalSlow': 5,
            'portalFast': 6,
            'sam': 7,
            'base': 8
        }
        feature_vector = [0] * len(object_mapping)
        if not cell:  # 检查单元格是否为空  
            return feature_vector  # 返回零特征  
        for obj in cell:
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] = 1
        return feature_vector

    features = []
    for row in grid:
        for cell in row:
            cell_feature = cell_to_feature(cell)
            features.extend(cell_feature)
    # print(features)
    return np.array(features)  # 返回NumPy数组

def extract_features_plus(observation):
    grid = observation
    
    object_mapping = {
        'wall': 1,
        'avatar': 2,
        'alien': 3,
        'bomb': 4,
        'base': 0
    }
    features = [[0, 0, 0, 0, 0] for _ in range(32)]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # print(grid[i][j])
            for obj in grid[i][j]:
                index = object_mapping.get(obj, -1)
                if index >= 0:
                    features[j][index] = 1

    features_array = np.array(features).flatten()  # 或者 np.ravel()
    #print(features)
    # print(features_array)
    return features_array
                
    
def extract_features_plus_plus(observation):
    grid = observation
    object_mapping = {
        'wall': 1,
        'avatar': 2,
        'alien': 3,
        'bomb': 4,
        'base': 0
    }
    features = [[0, 0, 0, 0, 0] for _ in range(32)]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            for obj in grid[i][j]:
                index = object_mapping.get(obj, -1)
                if index >= 0:
                    features[j][index] = 1
    features_array = np.array(features).flatten()
    return features_array
    

if __name__ == '__main__':
    observation = Load_game_records.load_game_records()[0][0]
    # print(observation)
    extract_features_plus(observation)
    # extract_features(observation)