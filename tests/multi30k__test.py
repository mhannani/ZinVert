from torch.utils.data import DataLoader
from src.data.Multi30k import Multi30k
from src.data.collate_fn import CollateFn
from src.data.Multi30k import CustomMulti30k


if __name__ == "__main__":
    custom_multi30k = CustomMulti30k()
    train, valid, test = custom_multi30k.extract_sets()
    print('DATASET SUMMARY')
    print('+++++++++++++++')
    print(f'+ Train test: {train.__len__()} sentences')
    print(f'+ Valid test: {valid.__len__()} sentences')
    print(f'+ Test test: {test.__len__()} sentences')
    print('+++++++++++++++++++++++++++++++++++++++++++')
    train_dataloader = DataLoader(train, batch_size=1, collate_fn=CollateFn())
    print('train test')
    for i, (src, target) in enumerate(train_dataloader):
        print('..', {i})
        print('src: ', src)
        print('target: ', target)
        break
