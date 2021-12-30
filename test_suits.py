from tests.vocabulary__test import test_vocabulary_test
from tests.encoder_att__test import encoder_att_test
from tests.encoder__test import encoder_test


if __name__ == "__main__":
    print('no att')
    encoder_test()
    print('with att')
    encoder_att_test()
