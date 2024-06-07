TRAIN_SPLIT = []
VAL_SPLIT = []
TEST_SPLIT = []

TRAIN_SPLIT += [
    'fish:top_fin:yes,mouth:bubbles,flipper:yes,tail:green', 
    'fish:top_fin:no,mouth:bubbles,flipper:yes,tail:green', 
    'fish:top_fin:yes,mouth:no,flipper:yes,tail:green', 
    'fish:top_fin:yes,mouth:bubbles,flipper:no,tail:green', 
    'fish:top_fin:yes,mouth:no,flipper:no,tail:green', 
    'fish:top_fin:no,mouth:no,flipper:no,tail:green', 
    'fish:top_fin:no,mouth:bubbles,flipper:yes,tail:blue', 
    'fish:top_fin:no,mouth:no,flipper:yes,tail:blue', 
    'fish:top_fin:yes,mouth:bubbles,flipper:no,tail:blue', 
    'fish:top_fin:yes,mouth:no,flipper:no,tail:blue', 
    'fish:top_fin:no,mouth:no,flipper:no,tail:blue',
    'fish:top_fin:no,mouth:bubbles,flipper:no,tail:green',
]
VAL_SPLIT += [
    'fish:top_fin:yes,mouth:no,flipper:yes,tail:green',
    'fish:top_fin:yes,mouth:no,flipper:no,tail:green',    
    'fish:top_fin:no,mouth:bubbles,flipper:no,tail:green',
    'fish:top_fin:no,mouth:bubbles,flipper:yes,tail:blue', 
]

TEST_SPLIT += [
    'fish:top_fin:yes,mouth:yes,flipper:no,tail:green', 
    'fish:top_fin:no,mouth:yes,flipper:no,tail:green', 
    'fish:top_fin:yes,mouth:yes,flipper:no,tail:blue', 
    'fish:top_fin:no,mouth:yes,flipper:no,tail:blue', 
    'fish:top_fin:yes,mouth:yes,flipper:yes,tail:green', 
    'fish:top_fin:no,mouth:yes,flipper:yes,tail:green', 
    'fish:top_fin:yes,mouth:yes,flipper:yes,tail:blue', 
    'fish:top_fin:no,mouth:yes,flipper:yes,tail:blue', 

    'fish:top_fin:yes,mouth:bubbles,flipper:yes,tail:blue',
    'fish:top_fin:no,mouth:bubbles,flipper:no,tail:blue',
    'fish:top_fin:yes,mouth:no,flipper:yes,tail:blue',
    'fish:top_fin:no,mouth:no,flipper:yes,tail:green', 
]

TRAIN_SPLIT += [
    'person:hat:regular,must:yes,tie:yes,eye:green', 
    'person:hat:no,must:yes,tie:yes,eye:green', 
    'person:hat:regular,must:yes,tie:no,eye:green', 
    'person:hat:no,must:yes,tie:no,eye:green',     
    'person:hat:regular,must:yes,tie:yes,eye:blue', 
    'person:hat:regular,must:no,tie:yes,eye:blue', 
    'person:hat:no,must:no,tie:yes,eye:blue', 
    'person:hat:no,must:yes,tie:no,eye:blue', 
    'person:hat:regular,must:no,tie:no,eye:blue', 
    'person:hat:no,must:no,tie:no,eye:blue',
    'person:hat:no,must:no,tie:yes,eye:green',
    'person:hat:no,must:no,tie:no,eye:green',    
]
VAL_SPLIT += [
    'person:hat:no,must:no,tie:no,eye:green',    
    'person:hat:regular,must:yes,tie:yes,eye:blue',
    'person:hat:no,must:yes,tie:no,eye:green',
    'person:hat:no,must:yes,tie:yes,eye:green', 
]

TEST_SPLIT += [
    'person:hat:pointy,must:yes,tie:yes,eye:green',
    'person:hat:pointy,must:no,tie:no,eye:blue',
    'person:hat:pointy,must:yes,tie:no,eye:blue',
    'person:hat:pointy,must:no,tie:yes,eye:blue',
    'person:hat:pointy,must:yes,tie:yes,eye:blue',
    'person:hat:pointy,must:no,tie:no,eye:green',
    'person:hat:pointy,must:yes,tie:no,eye:green',
    'person:hat:pointy,must:no,tie:yes,eye:green',

    'person:hat:regular,must:no,tie:yes,eye:green',
    'person:hat:no,must:yes,tie:yes,eye:blue',
    'person:hat:regular,must:yes,tie:no,eye:blue',
    'person:hat:regular,must:no,tie:no,eye:green', 
]

TRAIN_SPLIT += [
    'caterpillar:body:green,feet:circle,ant:yes,bluefeet:yes', 
    'caterpillar:body:red,feet:circle,ant:yes,bluefeet:yes', 
    'caterpillar:body:green,feet:triangle,ant:yes,bluefeet:yes', 
    'caterpillar:body:green,feet:circle,ant:no,bluefeet:yes', 
    'caterpillar:body:green,feet:triangle,ant:no,bluefeet:yes', 
    'caterpillar:body:red,feet:triangle,ant:no,bluefeet:yes', 
    'caterpillar:body:red,feet:circle,ant:yes,bluefeet:no', 
    'caterpillar:body:green,feet:triangle,ant:yes,bluefeet:no', 
    'caterpillar:body:red,feet:triangle,ant:yes,bluefeet:no', 
    'caterpillar:body:green,feet:circle,ant:no,bluefeet:no', 
    'caterpillar:body:red,feet:circle,ant:no,bluefeet:no', 
    'caterpillar:body:red,feet:triangle,ant:no,bluefeet:no', 
]
VAL_SPLIT += [
    'caterpillar:body:red,feet:triangle,ant:yes,bluefeet:no',
    'caterpillar:body:green,feet:circle,ant:no,bluefeet:no',
    'caterpillar:body:green,feet:triangle,ant:yes,bluefeet:yes',
    'caterpillar:body:green,feet:triangle,ant:no,bluefeet:yes', 
]
TEST_SPLIT += [
    'caterpillar:body:green,feet:triangle,ant:no,bluefeet:no',
    'caterpillar:body:green,feet:circle,ant:yes,bluefeet:no', 
    'caterpillar:body:red,feet:circle,ant:no,bluefeet:yes',
    'caterpillar:body:red,feet:triangle,ant:yes,bluefeet:yes', 
]

TRAIN_SPLIT += [
    'flower:petalcount:four,leaves:side,petalcolor:red,pot:yes', 
    'flower:petalcount:six,leaves:side,petalcolor:red,pot:yes', 
    'flower:petalcount:four,leaves:none,petalcolor:red,pot:yes', 
    'flower:petalcount:six,leaves:none,petalcolor:red,pot:yes', 
    'flower:petalcount:four,leaves:none,petalcolor:red,pot:no',
    'flower:petalcount:six,leaves:side,petalcolor:blue,pot:yes', 
    'flower:petalcount:six,leaves:none,petalcolor:blue,pot:yes', 
    'flower:petalcount:six,leaves:side,petalcolor:red,pot:no', 
    'flower:petalcount:six,leaves:none,petalcolor:red,pot:no', 
    'flower:petalcount:four,leaves:side,petalcolor:blue,pot:no', 
    'flower:petalcount:four,leaves:none,petalcolor:blue,pot:no', 
    'flower:petalcount:six,leaves:none,petalcolor:blue,pot:no', 
]
VAL_SPLIT += [
    'flower:petalcount:six,leaves:none,petalcolor:blue,pot:yes',
    'flower:petalcount:four,leaves:none,petalcolor:red,pot:yes',
    'flower:petalcount:six,leaves:side,petalcolor:blue,pot:yes',
    'flower:petalcount:four,leaves:side,petalcolor:blue,pot:no', 
]
TEST_SPLIT += [
    'flower:petalcount:four,leaves:double,petalcolor:red,pot:no', 
    'flower:petalcount:six,leaves:double,petalcolor:red,pot:no', 
    'flower:petalcount:four,leaves:double,petalcolor:blue,pot:no', 
    'flower:petalcount:six,leaves:double,petalcolor:blue,pot:no', 
    'flower:petalcount:four,leaves:double,petalcolor:blue,pot:yes', 
    'flower:petalcount:six,leaves:double,petalcolor:blue,pot:yes', 
    'flower:petalcount:four,leaves:double,petalcolor:red,pot:yes', 
    'flower:petalcount:six,leaves:double,petalcolor:red,pot:yes', 

    'flower:petalcount:four,leaves:none,petalcolor:blue,pot:yes',
    'flower:petalcount:four,leaves:side,petalcolor:red,pot:no', 
    'flower:petalcount:six,leaves:side,petalcolor:blue,pot:no',
    'flower:petalcount:four,leaves:side,petalcolor:blue,pot:yes', 

]

TRAIN_SPLIT += [
    'mushroom:stem:square,dots:ref,dotcolor:red,hang:yes', 
    'mushroom:stem:circle,dots:ref,dotcolor:red,hang:yes', 
    'mushroom:stem:square,dots:trans,dotcolor:red,hang:yes', 
    'mushroom:stem:square,dots:ref,dotcolor:blue,hang:yes', 
    'mushroom:stem:circle,dots:ref,dotcolor:blue,hang:yes', 
    'mushroom:stem:circle,dots:trans,dotcolor:blue,hang:yes', 
    'mushroom:stem:square,dots:ref,dotcolor:red,hang:no', 
    'mushroom:stem:square,dots:trans,dotcolor:red,hang:no', 
    'mushroom:stem:circle,dots:trans,dotcolor:red,hang:no', 
    'mushroom:stem:square,dots:ref,dotcolor:blue,hang:no', 
    'mushroom:stem:circle,dots:ref,dotcolor:blue,hang:no', 
    'mushroom:stem:circle,dots:trans,dotcolor:blue,hang:no', 
]
VAL_SPLIT += [
    'mushroom:stem:circle,dots:ref,dotcolor:red,hang:yes',
    'mushroom:stem:circle,dots:ref,dotcolor:blue,hang:yes',
    'mushroom:stem:square,dots:ref,dotcolor:red,hang:no',
    'mushroom:stem:circle,dots:trans,dotcolor:red,hang:no', 
]
TEST_SPLIT += [
    'mushroom:stem:circle,dots:ref,dotcolor:red,hang:no',
    'mushroom:stem:square,dots:trans,dotcolor:blue,hang:no',
    'mushroom:stem:circle,dots:trans,dotcolor:red,hang:yes',
    'mushroom:stem:square,dots:trans,dotcolor:blue,hang:yes', 
]

TRAIN_SPLIT += [
    'crab:arms:top_in,feet:yes,crabcolor:red,clawcolor:grey', 
    'crab:arms:side,feet:yes,crabcolor:red,clawcolor:grey', 
    'crab:arms:top_in,feet:no,crabcolor:red,clawcolor:grey', 
    'crab:arms:side,feet:no,crabcolor:red,clawcolor:grey', 
    'crab:arms:side,feet:yes,crabcolor:blue,clawcolor:grey',     
    'crab:arms:side,feet:no,crabcolor:blue,clawcolor:grey', 
    'crab:arms:top_in,feet:yes,crabcolor:red,clawcolor:match', 
    'crab:arms:top_in,feet:no,crabcolor:red,clawcolor:match', 
    'crab:arms:side,feet:no,crabcolor:red,clawcolor:match', 
    'crab:arms:side,feet:yes,crabcolor:blue,clawcolor:match', 
    'crab:arms:top_in,feet:no,crabcolor:blue,clawcolor:match', 
    'crab:arms:side,feet:no,crabcolor:blue,clawcolor:match', 
]
VAL_SPLIT += [
    'crab:arms:top_in,feet:no,crabcolor:red,clawcolor:grey',
    'crab:arms:side,feet:no,crabcolor:blue,clawcolor:grey',
    'crab:arms:side,feet:no,crabcolor:red,clawcolor:match',
    'crab:arms:top_in,feet:yes,crabcolor:red,clawcolor:match', 
]
TEST_SPLIT += [
    'crab:arms:top_out,feet:yes,crabcolor:red,clawcolor:grey',
    'crab:arms:top_out,feet:no,crabcolor:red,clawcolor:grey',
    'crab:arms:top_out,feet:yes,crabcolor:blue,clawcolor:grey',
    'crab:arms:top_out,feet:no,crabcolor:blue,clawcolor:grey',
    'crab:arms:top_out,feet:yes,crabcolor:red,clawcolor:match',
    'crab:arms:top_out,feet:no,crabcolor:red,clawcolor:match',
    'crab:arms:top_out,feet:yes,crabcolor:blue,clawcolor:match',
    'crab:arms:top_out,feet:no,crabcolor:blue,clawcolor:match',

    'crab:arms:top_in,feet:no,crabcolor:blue,clawcolor:grey', 
    'crab:arms:side,feet:yes,crabcolor:red,clawcolor:match',
    'crab:arms:top_in,feet:yes,crabcolor:blue,clawcolor:match',
    'crab:arms:top_in,feet:yes,crabcolor:blue,clawcolor:grey', 
    
]

TRAIN_SPLIT += [
    'cat:whisk:yes,mouth:yes,eyes:green,paws:yes', 
    'cat:whisk:no,mouth:yes,eyes:green,paws:yes', 
    'cat:whisk:yes,mouth:no,eyes:green,paws:yes', 
    'cat:whisk:yes,mouth:yes,eyes:blue,paws:yes', 
    'cat:whisk:no,mouth:yes,eyes:blue,paws:yes', 
    'cat:whisk:no,mouth:no,eyes:blue,paws:yes', 
    'cat:whisk:yes,mouth:yes,eyes:green,paws:no', 
    'cat:whisk:yes,mouth:no,eyes:green,paws:no', 
    'cat:whisk:no,mouth:no,eyes:green,paws:no', 
    'cat:whisk:no,mouth:yes,eyes:blue,paws:no', 
    'cat:whisk:yes,mouth:no,eyes:blue,paws:no', 
    'cat:whisk:no,mouth:no,eyes:blue,paws:no', 
]
VAL_SPLIT += [
    'cat:whisk:no,mouth:no,eyes:blue,paws:yes',
    'cat:whisk:yes,mouth:no,eyes:green,paws:yes',
    'cat:whisk:yes,mouth:yes,eyes:green,paws:no',
    'cat:whisk:yes,mouth:no,eyes:blue,paws:no', 
]
TEST_SPLIT += [
    'cat:whisk:yes,mouth:yes,eyes:blue,paws:no',
    'cat:whisk:no,mouth:yes,eyes:green,paws:no',
    'cat:whisk:yes,mouth:no,eyes:blue,paws:yes',
    'cat:whisk:no,mouth:no,eyes:green,paws:yes', 
]

TEST_SPLIT += [
    'turtle:spot:one,tail:circle,legs:yes,spotcolor:red', 
    'turtle:spot:multi,tail:circle,legs:yes,spotcolor:red', 
    'turtle:spot:one,tail:circle,legs:no,spotcolor:red', 
    'turtle:spot:multi,tail:circle,legs:no,spotcolor:red', 
    'turtle:spot:one,tail:triangle,legs:no,spotcolor:red', 
    'turtle:spot:multi,tail:triangle,legs:no,spotcolor:red', 
    'turtle:spot:one,tail:circle,legs:yes,spotcolor:blue', 
    'turtle:spot:multi,tail:circle,legs:yes,spotcolor:blue', 
    'turtle:spot:multi,tail:triangle,legs:yes,spotcolor:blue', 
    'turtle:spot:one,tail:circle,legs:no,spotcolor:blue',     
    'turtle:spot:one,tail:triangle,legs:no,spotcolor:blue', 
    'turtle:spot:multi,tail:triangle,legs:no,spotcolor:blue', 
    'turtle:spot:one,tail:triangle,legs:yes,spotcolor:blue',
    'turtle:spot:one,tail:triangle,legs:yes,spotcolor:red',
    'turtle:spot:multi,tail:triangle,legs:yes,spotcolor:red',
    'turtle:spot:multi,tail:circle,legs:no,spotcolor:blue', 
]

TRAIN_SPLIT += [
    'mouse:feet:yes,tail:yes,inner:yes,eyes:green', 
    'mouse:feet:no,tail:yes,inner:yes,eyes:green', 
    'mouse:feet:yes,tail:no,inner:yes,eyes:green', 
    'mouse:feet:yes,tail:yes,inner:no,eyes:green', 
    'mouse:feet:no,tail:yes,inner:no,eyes:green', 
    'mouse:feet:no,tail:no,inner:no,eyes:green', 
    'mouse:feet:yes,tail:yes,inner:yes,eyes:blue', 
    'mouse:feet:yes,tail:no,inner:yes,eyes:blue', 
    'mouse:feet:no,tail:no,inner:yes,eyes:blue', 
    'mouse:feet:no,tail:yes,inner:no,eyes:blue', 
    'mouse:feet:yes,tail:no,inner:no,eyes:blue', 
    'mouse:feet:no,tail:no,inner:no,eyes:blue', 
]
VAL_SPLIT += [
    'mouse:feet:yes,tail:no,inner:yes,eyes:green',
    'mouse:feet:yes,tail:no,inner:yes,eyes:blue', 
    'mouse:feet:no,tail:yes,inner:no,eyes:green',
    'mouse:feet:no,tail:yes,inner:no,eyes:blue', 
]
TEST_SPLIT += [
    'mouse:feet:yes,tail:yes,inner:no,eyes:blue',
    'mouse:feet:no,tail:yes,inner:yes,eyes:blue',
    'mouse:feet:no,tail:no,inner:yes,eyes:green',
    'mouse:feet:yes,tail:no,inner:no,eyes:green', 
]

TRAIN_SPLIT += [
    'ladybug:dotnum:four,arms:yes,eyes:yes,centerdot:yes', 
    'ladybug:dotnum:six,arms:yes,eyes:yes,centerdot:yes', 
    'ladybug:dotnum:four,arms:no,eyes:yes,centerdot:yes', 
    'ladybug:dotnum:six,arms:no,eyes:yes,centerdot:yes', 
    'ladybug:dotnum:four,arms:yes,eyes:no,centerdot:yes', 
    'ladybug:dotnum:six,arms:yes,eyes:no,centerdot:yes', 
    'ladybug:dotnum:six,arms:no,eyes:no,centerdot:yes', 
    'ladybug:dotnum:four,arms:yes,eyes:yes,centerdot:no', 
    'ladybug:dotnum:six,arms:yes,eyes:yes,centerdot:no', 
    'ladybug:dotnum:four,arms:yes,eyes:no,centerdot:no', 
    'ladybug:dotnum:four,arms:no,eyes:no,centerdot:no', 
    'ladybug:dotnum:six,arms:no,eyes:no,centerdot:no', 
]
VAL_SPLIT += [
    'ladybug:dotnum:six,arms:yes,eyes:yes,centerdot:no',
    'ladybug:dotnum:four,arms:yes,eyes:yes,centerdot:no',
    'ladybug:dotnum:six,arms:no,eyes:no,centerdot:yes',
    'ladybug:dotnum:six,arms:no,eyes:yes,centerdot:yes', 
]
TEST_SPLIT += [
    'ladybug:dotnum:four,arms:no,eyes:no,centerdot:yes',
    'ladybug:dotnum:four,arms:no,eyes:yes,centerdot:no',
    'ladybug:dotnum:six,arms:no,eyes:yes,centerdot:no',
    'ladybug:dotnum:six,arms:yes,eyes:no,centerdot:no', 
]

TRAIN_SPLIT += [
    'fridge:top:yes,half:yes,ice:yes,handles:yes', 
    'fridge:top:no,half:yes,ice:yes,handles:yes', 
    'fridge:top:yes,half:no,ice:yes,handles:yes', 
    'fridge:top:no,half:no,ice:yes,handles:yes', 
    'fridge:top:no,half:yes,ice:no,handles:yes', 
    'fridge:top:yes,half:no,ice:no,handles:yes', 
    'fridge:top:yes,half:yes,ice:yes,handles:no', 
    'fridge:top:yes,half:no,ice:yes,handles:no', 
    'fridge:top:no,half:no,ice:yes,handles:no', 
    'fridge:top:no,half:yes,ice:no,handles:no', 
    'fridge:top:yes,half:no,ice:no,handles:no', 
    'fridge:top:no,half:no,ice:no,handles:no', 
]
VAL_SPLIT += [
    'fridge:top:no,half:no,ice:yes,handles:yes',
    'fridge:top:yes,half:no,ice:no,handles:yes',
    'fridge:top:yes,half:no,ice:yes,handles:no',
    'fridge:top:yes,half:yes,ice:yes,handles:no', 
]
TEST_SPLIT += [
    'fridge:top:yes,half:yes,ice:no,handles:yes',
    'fridge:top:no,half:no,ice:no,handles:yes',
    'fridge:top:no,half:yes,ice:yes,handles:no',
    'fridge:top:yes,half:yes,ice:no,handles:no', 
]

TRAIN_SPLIT += [
    'microwave:handle:yes,feet:yes,tray:yes,button:triple', 
    'microwave:handle:no,feet:yes,tray:yes,button:triple', 
    'microwave:handle:yes,feet:no,tray:yes,button:triple', 
    'microwave:handle:yes,feet:yes,tray:no,button:triple', 
    'microwave:handle:yes,feet:no,tray:no,button:triple', 
    'microwave:handle:no,feet:no,tray:no,button:triple', 
    'microwave:handle:no,feet:yes,tray:yes,button:single', 
    'microwave:handle:yes,feet:no,tray:yes,button:single', 
    'microwave:handle:no,feet:no,tray:yes,button:single', 
    'microwave:handle:no,feet:yes,tray:no,button:single', 
    'microwave:handle:yes,feet:no,tray:no,button:single', 
    'microwave:handle:no,feet:no,tray:no,button:single', 
]
VAL_SPLIT += [
    'microwave:handle:yes,feet:no,tray:yes,button:single',
    'microwave:handle:yes,feet:yes,tray:no,button:triple', 
    'microwave:handle:no,feet:no,tray:yes,button:single',
    'microwave:handle:no,feet:yes,tray:yes,button:triple', 
]
TEST_SPLIT += [
    'microwave:handle:yes,feet:yes,tray:yes,button:double', 
    'microwave:handle:no,feet:yes,tray:yes,button:double', 
    'microwave:handle:yes,feet:no,tray:yes,button:double', 
    'microwave:handle:no,feet:no,tray:yes,button:double', 
    'microwave:handle:yes,feet:yes,tray:no,button:double', 
    'microwave:handle:no,feet:yes,tray:no,button:double', 
    'microwave:handle:yes,feet:no,tray:no,button:double', 
    'microwave:handle:no,feet:no,tray:no,button:double',

    'microwave:handle:yes,feet:yes,tray:no,button:single',
    'microwave:handle:yes,feet:yes,tray:yes,button:single',
    'microwave:handle:no,feet:yes,tray:no,button:triple',
    'microwave:handle:no,feet:no,tray:yes,button:triple', 
]

TRAIN_SPLIT += [
    'clock:frame:yes,feet:yes,bells:yes,hands:one', 
    'clock:frame:no,feet:yes,bells:yes,hands:one', 
    'clock:frame:no,feet:no,bells:yes,hands:one', 
    'clock:frame:yes,feet:yes,bells:no,hands:one', 
    'clock:frame:no,feet:yes,bells:no,hands:one', 
    'clock:frame:no,feet:no,bells:no,hands:one', 
    'clock:frame:yes,feet:yes,bells:yes,hands:diff', 
    'clock:frame:yes,feet:no,bells:yes,hands:diff', 
    'clock:frame:no,feet:no,bells:yes,hands:diff', 
    'clock:frame:no,feet:yes,bells:no,hands:diff', 
    'clock:frame:yes,feet:no,bells:no,hands:diff', 
    'clock:frame:no,feet:no,bells:no,hands:diff', 
]
VAL_SPLIT += [
    'clock:frame:no,feet:yes,bells:no,hands:diff',
    'clock:frame:no,feet:yes,bells:no,hands:one',
    'clock:frame:yes,feet:no,bells:no,hands:diff',
    'clock:frame:no,feet:yes,bells:yes,hands:one', 
]
TEST_SPLIT += [
    'clock:frame:yes,feet:yes,bells:yes,hands:same', 
    'clock:frame:no,feet:yes,bells:yes,hands:same', 
    'clock:frame:yes,feet:no,bells:yes,hands:same', 
    'clock:frame:no,feet:no,bells:yes,hands:same', 
    'clock:frame:yes,feet:yes,bells:no,hands:same', 
    'clock:frame:no,feet:yes,bells:no,hands:same', 
    'clock:frame:yes,feet:no,bells:no,hands:same', 
    'clock:frame:no,feet:no,bells:no,hands:same', 

    'clock:frame:yes,feet:no,bells:no,hands:one',
    'clock:frame:no,feet:yes,bells:yes,hands:diff',
    'clock:frame:yes,feet:yes,bells:no,hands:diff',
    'clock:frame:yes,feet:no,bells:yes,hands:one', 
]

TRAIN_SPLIT += [
    'car:window:yes,top:yes,hub:yes,road:yes', 
    'car:window:no,top:yes,hub:yes,road:yes', 
    'car:window:yes,top:no,hub:yes,road:yes', 
    'car:window:yes,top:yes,hub:no,road:yes', 
    'car:window:no,top:yes,hub:no,road:yes', 
    'car:window:no,top:no,hub:no,road:yes', 
    'car:window:no,top:yes,hub:yes,road:no', 
    'car:window:yes,top:no,hub:yes,road:no', 
    'car:window:no,top:no,hub:yes,road:no', 
    'car:window:no,top:yes,hub:no,road:no', 
    'car:window:yes,top:no,hub:no,road:no', 
    'car:window:no,top:no,hub:no,road:no', 
]
VAL_SPLIT += [
   'car:window:no,top:yes,hub:no,road:no',
    'car:window:yes,top:no,hub:yes,road:no',
    'car:window:yes,top:no,hub:yes,road:yes',
    'car:window:no,top:no,hub:no,road:yes', 
]
TEST_SPLIT += [
    'car:window:yes,top:yes,hub:yes,road:no',
    'car:window:yes,top:yes,hub:no,road:no',
    'car:window:no,top:no,hub:yes,road:yes',
    'car:window:yes,top:no,hub:no,road:yes', 
]

TRAIN_SPLIT += [
    'plane:wing:single,engine:body,prop:yes,body:color', 
    'plane:wing:double,engine:body,prop:yes,body:color', 
    'plane:wing:single,engine:no,prop:yes,body:color', 
    'plane:wing:double,engine:no,prop:yes,body:color', 
    'plane:wing:double,engine:body,prop:no,body:color', 
    'plane:wing:single,engine:no,prop:no,body:color', 
    'plane:wing:single,engine:body,prop:yes,body:grey', 
    'plane:wing:double,engine:body,prop:yes,body:grey', 
    'plane:wing:double,engine:no,prop:yes,body:grey', 
    'plane:wing:single,engine:body,prop:no,body:grey', 
    'plane:wing:double,engine:body,prop:no,body:grey', 
    'plane:wing:double,engine:no,prop:no,body:grey', 
]
VAL_SPLIT += [
    'plane:wing:double,engine:no,prop:yes,body:grey',
    'plane:wing:single,engine:no,prop:no,body:color',
    'plane:wing:double,engine:body,prop:yes,body:color',
    'plane:wing:double,engine:body,prop:no,body:grey', 
]
TEST_SPLIT += [
    'plane:wing:single,engine:wing,prop:yes,body:color', 
    'plane:wing:double,engine:wing,prop:yes,body:color', 
    'plane:wing:single,engine:wing,prop:no,body:color', 
    'plane:wing:double,engine:wing,prop:no,body:color', 
    'plane:wing:single,engine:wing,prop:yes,body:grey', 
    'plane:wing:double,engine:wing,prop:yes,body:grey', 
    'plane:wing:single,engine:wing,prop:no,body:grey', 
    'plane:wing:double,engine:wing,prop:no,body:grey', 

    'plane:wing:single,engine:no,prop:yes,body:grey',
    'plane:wing:double,engine:no,prop:no,body:color',
    'plane:wing:single,engine:body,prop:no,body:color',
    'plane:wing:single,engine:no,prop:no,body:grey', 
]

TRAIN_SPLIT += [
    'horiz_back:slats:trans,top_bar:yes,bot_bar:yes,frame:grey', 
    'horiz_back:slats:ref,top_bar:yes,bot_bar:yes,frame:grey', 
    'horiz_back:slats:ref,top_bar:no,bot_bar:yes,frame:grey', 
    'horiz_back:slats:trans,top_bar:yes,bot_bar:no,frame:grey', 
    'horiz_back:slats:trans,top_bar:no,bot_bar:no,frame:grey', 
    'horiz_back:slats:ref,top_bar:no,bot_bar:no,frame:grey', 
    'horiz_back:slats:trans,top_bar:yes,bot_bar:yes,frame:match', 
    'horiz_back:slats:ref,top_bar:yes,bot_bar:yes,frame:match',     
    'horiz_back:slats:ref,top_bar:no,bot_bar:yes,frame:match', 
    'horiz_back:slats:trans,top_bar:yes,bot_bar:no,frame:match',     
    'horiz_back:slats:trans,top_bar:no,bot_bar:no,frame:match', 
    'horiz_back:slats:ref,top_bar:no,bot_bar:no,frame:match', 
]
VAL_SPLIT += [
    'horiz_back:slats:trans,top_bar:yes,bot_bar:yes,frame:match',
    'horiz_back:slats:ref,top_bar:no,bot_bar:yes,frame:match',
    'horiz_back:slats:trans,top_bar:yes,bot_bar:no,frame:grey',
    'horiz_back:slats:ref,top_bar:yes,bot_bar:yes,frame:grey', 
]
TEST_SPLIT += [
    'horiz_back:slats:trans,top_bar:no,bot_bar:yes,frame:grey',
    'horiz_back:slats:ref,top_bar:yes,bot_bar:no,frame:grey',
    'horiz_back:slats:trans,top_bar:no,bot_bar:yes,frame:match',
    'horiz_back:slats:ref,top_bar:yes,bot_bar:no,frame:match', 
]

TRAIN_SPLIT += [
    'house:chimney:no,window:single,bushes:yes,mcolor:yes', 
    'house:chimney:yes,window:double,bushes:yes,mcolor:yes', 
    'house:chimney:no,window:double,bushes:yes,mcolor:yes', 
    'house:chimney:no,window:single,bushes:no,mcolor:yes', 
    'house:chimney:yes,window:double,bushes:no,mcolor:yes', 
    'house:chimney:no,window:double,bushes:no,mcolor:yes', 
    'house:chimney:no,window:single,bushes:yes,mcolor:no', 
    'house:chimney:yes,window:double,bushes:yes,mcolor:no',     
    'house:chimney:yes,window:single,bushes:no,mcolor:no', 
    'house:chimney:no,window:single,bushes:no,mcolor:no', 
    'house:chimney:yes,window:double,bushes:no,mcolor:no', 
    'house:chimney:no,window:double,bushes:no,mcolor:no', 
]
VAL_SPLIT += [
    'house:chimney:no,window:single,bushes:yes,mcolor:yes',
    'house:chimney:yes,window:double,bushes:no,mcolor:yes',
    'house:chimney:yes,window:double,bushes:yes,mcolor:no',
    'house:chimney:yes,window:double,bushes:no,mcolor:no', 
]
TEST_SPLIT += [
    'house:chimney:yes,window:single,bushes:yes,mcolor:yes',
    'house:chimney:yes,window:single,bushes:yes,mcolor:no',
    'house:chimney:no,window:double,bushes:yes,mcolor:no',
    'house:chimney:yes,window:single,bushes:no,mcolor:yes', 
]

TRAIN_SPLIT += [
    'side_chair:base:pedestal,facing:right,armrest:yes,color:diff', 
    'side_chair:base:regular,facing:right,armrest:yes,color:diff',     
    'side_chair:base:pedestal,facing:right,armrest:no,color:diff', 
    'side_chair:base:regular,facing:right,armrest:no,color:diff', 
    'side_chair:base:regular,facing:left,armrest:no,color:diff', 
    'side_chair:base:pedestal,facing:right,armrest:yes,color:same', 
    'side_chair:base:regular,facing:right,armrest:yes,color:same', 
    'side_chair:base:pedestal,facing:left,armrest:yes,color:same', 
    'side_chair:base:regular,facing:left,armrest:yes,color:same', 
    'side_chair:base:regular,facing:right,armrest:no,color:same', 
    'side_chair:base:pedestal,facing:left,armrest:no,color:same', 
    'side_chair:base:regular,facing:left,armrest:no,color:same', 
]
VAL_SPLIT += [
    'side_chair:base:pedestal,facing:left,armrest:no,color:same',
    'side_chair:base:pedestal,facing:left,armrest:yes,color:same',
    'side_chair:base:regular,facing:right,armrest:no,color:diff',
    'side_chair:base:regular,facing:right,armrest:yes,color:diff',     
]
TEST_SPLIT += [
    'side_chair:base:pedestal,facing:left,armrest:yes,color:diff',
    'side_chair:base:pedestal,facing:right,armrest:no,color:same',
    'side_chair:base:pedestal,facing:left,armrest:no,color:diff',
    'side_chair:base:regular,facing:left,armrest:yes,color:diff', 
]

TRAIN_SPLIT += [
    'table:base:regular,leaf:right,top:yes,knob:yes', 
    'table:base:pedestal,leaf:right,top:yes,knob:yes', 
    'table:base:regular,leaf:no,top:yes,knob:yes', 
    'table:base:regular,leaf:no,top:no,knob:yes', 
    'table:base:pedestal,leaf:no,top:no,knob:yes', 
    'table:base:regular,leaf:right,top:yes,knob:no', 
    'table:base:pedestal,leaf:right,top:yes,knob:no', 
    'table:base:regular,leaf:no,top:yes,knob:no', 
    'table:base:pedestal,leaf:no,top:yes,knob:no', 
    'table:base:regular,leaf:right,top:no,knob:no', 
    'table:base:pedestal,leaf:right,top:no,knob:no', 
    'table:base:pedestal,leaf:no,top:no,knob:no', 
]
VAL_SPLIT += [
    'table:base:pedestal,leaf:right,top:no,knob:no',
    'table:base:pedestal,leaf:no,top:yes,knob:no',
    'table:base:pedestal,leaf:right,top:yes,knob:no',
    'table:base:regular,leaf:no,top:yes,knob:yes', 
]
TEST_SPLIT += [
    'table:base:regular,leaf:left,top:yes,knob:yes', 
    'table:base:pedestal,leaf:left,top:yes,knob:yes', 
    'table:base:regular,leaf:left,top:no,knob:yes', 
    'table:base:pedestal,leaf:left,top:no,knob:yes', 
    'table:base:regular,leaf:left,top:yes,knob:no', 
    'table:base:pedestal,leaf:left,top:yes,knob:no', 
    'table:base:regular,leaf:left,top:no,knob:no', 
    'table:base:pedestal,leaf:left,top:no,knob:no', 

    'table:base:regular,leaf:right,top:no,knob:yes',
    'table:base:pedestal,leaf:right,top:no,knob:yes',
    'table:base:pedestal,leaf:no,top:yes,knob:yes',
    'table:base:regular,leaf:no,top:no,knob:no', 
]

TEST_SPLIT += [
    'bookshelf:feet:yes,sides:yes,frame:red,shelves:trans', 
    'bookshelf:feet:no,sides:yes,frame:red,shelves:trans', 
    'bookshelf:feet:yes,sides:no,frame:red,shelves:trans', 
    'bookshelf:feet:yes,sides:yes,frame:match,shelves:trans', 
    'bookshelf:feet:yes,sides:no,frame:match,shelves:trans', 
    'bookshelf:feet:no,sides:no,frame:match,shelves:trans', 
    'bookshelf:feet:no,sides:yes,frame:red,shelves:ref', 
    'bookshelf:feet:yes,sides:no,frame:red,shelves:ref',  
    'bookshelf:feet:yes,sides:yes,frame:match,shelves:ref', 
    'bookshelf:feet:no,sides:yes,frame:match,shelves:ref', 
    'bookshelf:feet:yes,sides:no,frame:match,shelves:ref', 
    'bookshelf:feet:no,sides:no,frame:match,shelves:ref',
    'bookshelf:feet:yes,sides:yes,frame:red,shelves:ref',
    'bookshelf:feet:no,sides:yes,frame:match,shelves:trans', 
    'bookshelf:feet:no,sides:no,frame:red,shelves:ref',
    'bookshelf:feet:no,sides:no,frame:red,shelves:trans', 
]
