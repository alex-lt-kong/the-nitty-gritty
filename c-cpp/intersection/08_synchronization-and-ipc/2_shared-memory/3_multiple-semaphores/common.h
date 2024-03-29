#define SEM_INITIAL_VALUE 1
#define SEM_COUNT 128

#define SHM_SIZE 4096
#define SHM_NAME "/shm.me"

#define PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)

const char sem_names[SEM_COUNT][32] = {
    "MySemaphore000",
    "MySemaphore001",
    "MySemaphore002",
    "MySemaphore003",
    "MySemaphore004",
    "MySemaphore005",
    "MySemaphore006",
    "MySemaphore007",
    "MySemaphore008",
    "MySemaphore009",
    "MySemaphore010",
    "MySemaphore011",
    "MySemaphore012",
    "MySemaphore013",
    "MySemaphore014",
    "MySemaphore015",
    "MySemaphore016",
    "MySemaphore017",
    "MySemaphore018",
    "MySemaphore019",
    "MySemaphore020",
    "MySemaphore021",
    "MySemaphore022",
    "MySemaphore023",
    "MySemaphore024",
    "MySemaphore025",
    "MySemaphore026",
    "MySemaphore027",
    "MySemaphore028",
    "MySemaphore029",
    "MySemaphore030",
    "MySemaphore031",
    "MySemaphore032",
    "MySemaphore033",
    "MySemaphore034",
    "MySemaphore035",
    "MySemaphore036",
    "MySemaphore037",
    "MySemaphore038",
    "MySemaphore039",
    "MySemaphore040",
    "MySemaphore041",
    "MySemaphore042",
    "MySemaphore043",
    "MySemaphore044",
    "MySemaphore045",
    "MySemaphore046",
    "MySemaphore047",
    "MySemaphore048",
    "MySemaphore049",
    "MySemaphore050",
    "MySemaphore051",
    "MySemaphore052",
    "MySemaphore053",
    "MySemaphore054",
    "MySemaphore055",
    "MySemaphore056",
    "MySemaphore057",
    "MySemaphore058",
    "MySemaphore059",
    "MySemaphore060",
    "MySemaphore061",
    "MySemaphore062",
    "MySemaphore063",
    "MySemaphore064",
    "MySemaphore065",
    "MySemaphore066",
    "MySemaphore067",
    "MySemaphore068",
    "MySemaphore069",
    "MySemaphore070",
    "MySemaphore071",
    "MySemaphore072",
    "MySemaphore073",
    "MySemaphore074",
    "MySemaphore075",
    "MySemaphore076",
    "MySemaphore077",
    "MySemaphore078",
    "MySemaphore079",
    "MySemaphore080",
    "MySemaphore081",
    "MySemaphore082",
    "MySemaphore083",
    "MySemaphore084",
    "MySemaphore085",
    "MySemaphore086",
    "MySemaphore087",
    "MySemaphore088",
    "MySemaphore089",
    "MySemaphore090",
    "MySemaphore091",
    "MySemaphore092",
    "MySemaphore093",
    "MySemaphore094",
    "MySemaphore095",
    "MySemaphore096",
    "MySemaphore097",
    "MySemaphore098",
    "MySemaphore099",
    "MySemaphore100",
    "MySemaphore101",
    "MySemaphore102",
    "MySemaphore103",
    "MySemaphore104",
    "MySemaphore105",
    "MySemaphore106",
    "MySemaphore107",
    "MySemaphore108",
    "MySemaphore109",
    "MySemaphore110",
    "MySemaphore111",
    "MySemaphore112",
    "MySemaphore113",
    "MySemaphore114",
    "MySemaphore115",
    "MySemaphore116",
    "MySemaphore117",
    "MySemaphore118",
    "MySemaphore119",
    "MySemaphore120",
    "MySemaphore121",
    "MySemaphore122",
    "MySemaphore123",
    "MySemaphore124",
    "MySemaphore125",
    "MySemaphore126",
    "MySemaphore127"
};
// o:wr, g:wr, i.e., 0660