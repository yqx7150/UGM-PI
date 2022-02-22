import sys,argparse
import os
import ugm_pi.main_SIAT as ugm_pi
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ugm_pi', help='The model to use as a prior')
    parsed,sys.argv = parser.parse_known_args()
    sys.argv.insert(0,parsed.model)

    print(parsed)

    if parsed.model == 'ugm_pi':
        ugm_pi.main()                       ################
    elif parsed.model == 'glow':
        glow.main()
    else:
        print('Unknown model \'{}\': please select a model from the list: {}'.format(parsed.model, ['ugm_pi','glow']))
