use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

#[allow(unused)]
pub mod splay_tree {
    use std::{cell::UnsafeCell, cmp::Ordering, mem};

    #[derive(Debug, Clone, Copy)]
    enum Branch {
        Left = 0,
        Right = 1,
    }

    impl Branch {
        fn inv(self) -> Self {
            match self {
                Branch::Left => Branch::Right,
                Branch::Right => Branch::Left,
            }
        }
    }

    #[derive(Debug)]
    struct Node<K> {
        key: K,
        children: [Option<Box<Node<K>>>; 2],
    }

    impl<K: Ord> Node<K> {
        fn new(key: K) -> Self {
            Self {
                key,
                children: [None, None],
            }
        }

        fn rotate(self: &mut Box<Self>, branch: Branch) {
            let mut sub: Box<Self> = self.children[branch.inv() as usize].take().unwrap();
            self.children[branch.inv() as usize] = sub.children[branch as usize].take();
            std::mem::swap(self, &mut sub);
            self.children[branch as usize] = Some(sub);
        }

        fn splay(self: &mut Box<Self>, key: K) {
            let mut new_children = [None, None];
            let mut cursors = unsafe {
                [
                    &mut *(&mut new_children[0] as *mut _),
                    &mut *(&mut new_children[1] as *mut _),
                ]
            };

            loop {
                let branch = match key.cmp(&self.key) {
                    Ordering::Equal => break,
                    Ordering::Less => Branch::Left,
                    Ordering::Greater => Branch::Right,
                };

                let (mut cursor, mut cursor_inv) = cursors.split_at_mut(1);
                if matches!(branch, Branch::Left) {
                    mem::swap(&mut cursor, &mut cursor_inv);
                }
                let (mut cursor, mut cursor_inv) = (&mut cursor[0], &mut cursor_inv[0]);

                match self.children[branch as usize].take() {
                    None => break,
                    Some(child) => {
                        if key.cmp(&child.key) == Ordering::Less {
                            self.rotate(branch.inv());
                            if self.children[branch as usize].is_none() {
                                break;
                            }
                        }

                        **cursor_inv = Some(mem::replace(self, child));
                        *cursor_inv = &mut cursor_inv.as_mut().unwrap().children[branch as usize];
                    }
                }
            }

            mem::swap(cursors[0], &mut self.children[0]);
            mem::swap(cursors[1], &mut self.children[1]);
            self.children = new_children;
        }
        /*
                splay* Splay(int key, splay* root)
        {
            if(!root)
                return NULL;
            splay header;
            /* header.rchild points to L tree; header.lchild points to R Tree */
            header.lchild = header.rchild = NULL;
            splay* LeftTreeMax = &header;
            splay* RightTreeMin = &header;

            /* loop until root->lchild == NULL || root->rchild == NULL; then break!
               (or when find the key, break too.)
             The zig/zag mode would only happen when cannot find key and will reach
             null on one side after RR or LL Rotation.
             */
            while(1)
            {
                if(key < root->key)
                {
                    if(!root->lchild)
                        break;
                    if(key < root->lchild->key)
                    {
                        root = RR_Rotate(root); /* only zig-zig mode need to rotate once,
                                                   because zig-zag mode is handled as zig
                                                   mode, which doesn't require rotate,
                                                   just linking it to R Tree */
                        if(!root->lchild)
                            break;
                    }
                    /* Link to R Tree */
                    RightTreeMin->lchild = root;
                    RightTreeMin = RightTreeMin->lchild;
                    root = root->lchild;
                    RightTreeMin->lchild = NULL;
                }
                else if(key > root->key)
                {
                    if(!root->rchild)
                        break;
                    if(key > root->rchild->key)
                    {
                        root = LL_Rotate(root);/* only zag-zag mode need to rotate once,
                                                  because zag-zig mode is handled as zag
                                                  mode, which doesn't require rotate,
                                                  just linking it to L Tree */
                        if(!root->rchild)
                            break;
                    }
                    /* Link to L Tree */
                    LeftTreeMax->rchild = root;
                    LeftTreeMax = LeftTreeMax->rchild;
                    root = root->rchild;
                    LeftTreeMax->rchild = NULL;
                }
                else
                    break;
            }
            /* assemble L Tree, Middle Tree and R tree together */
            LeftTreeMax->rchild = root->lchild;
            RightTreeMin->lchild = root->rchild;
            root->lchild = header.rchild;
            root->rchild = header.lchild;

            return root;
        }
                * */
    }

    struct SplayTree<K> {
        root: UnsafeCell<Option<Box<Node<K>>>>,
    }

    impl<K: Ord> SplayTree<K> {
        //
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
}
