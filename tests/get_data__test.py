from src.data.get_data import get_data

if __name__ == "__main__":
    tr = get_data(2, split='train')
    for i in tr:
        print(i)
        print('===')
