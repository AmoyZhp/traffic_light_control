ENV="1x1"
TRAINER="IQL"
REPLAY_BUFFER="Common"
EPISODE=2
BATCH_SIZE=2
python main.py --env $ENV --trainer $TRAINER --episodes $EPISODE --batch_size $BATCH_SIZE --replay_buffer $REPLAY_BUFFER