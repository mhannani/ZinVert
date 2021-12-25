from src.data.get_data import get_data

if __name__ == "__main__":
    tr = get_data(2, split='train')
    for epoch in range(2):
        # loop through batches
        for i, (src, tgt) in enumerate(tr):
            print(i)
