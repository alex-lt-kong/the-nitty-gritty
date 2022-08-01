#include <stdlib.h>
#include <string.h>

/* finalization is not done here, memory will leak. */

struct dict_item { /* table entry: */
    struct dict_item *next; /* next item in a linked list of items with the SAME hash value */
    char *key; /* defined name */
    char *value; /* replacement text */
};

#define HASHSIZE 101
static struct dict_item *hashtab[HASHSIZE]; /* pointer table */

/* hash: form hash value for string s */
unsigned hash(char *s)
{
    // This simple hash algorithm is from C Programming Language by Brian W. Kernighan and Dennis M. Ritchie
    // It is sometimes referred as "the K&R C hashing algorithm"
    // https://stackoverflow.com/questions/4384359/quick-way-to-implement-dictionary-in-c

    unsigned hashval;
    for (hashval = 0; *s != '\0'; s++) {
      hashval = *s + 31 * hashval;
    }
    return hashval % HASHSIZE;
}

/* lookup: look for s in hashtab */
struct dict_item *lookup(char *s)
{
    struct dict_item *np;
    for (np = hashtab[hash(s)]; np != NULL; np = np->next) {
    /* It is a bit confusing on why we need to follow the linked list. The reason is that, the linked list is NOT
    used to link all the element in the dictionary. Instead, it is only used to link the elements with the same hashval, 
    i.e., the elements whose names cause hash collision
    */
        if (strcmp(s, np->key) == 0) {
            return np; /* found */
        }
    }
    return NULL; /* not found */
}

/* install: put (key, value) in hashtab */
struct dict_item *install(char *key, char *value)
{
    struct dict_item *np;
    unsigned hashval;
    if ((np = lookup(key)) == NULL) { /* not found */
        np = (struct dict_item *) malloc(sizeof(*np));
        if (np == NULL || (np->key = strdup(key)) == NULL)
          return NULL;
        hashval = hash(key);
        // we prepend the new node to the linked list of elements with the same hash value.
        np->next = hashtab[hashval];
        hashtab[hashval] = np;
    } else {/* already there */
        free((void *) np->value); /*free previous value */
    }
    if ((np->value = strdup(value)) == NULL)
       return NULL;
    return np;
}
