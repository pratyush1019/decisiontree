#lang racket

(require 2htdp/batch-io)

(require "decision_functions.rkt")
;(require "testdata.rkt")


;input dataset
(provide toytrain)
(define toytrain "../data/toy_train.csv")

(provide titanictrain)
(define titanictrain "../data/titanic_train.csv")
;(define toytrain "../data/toy_train.csv")


(provide mushroomtrain)
(define mushroomtrain "../data/mushrooms_train.csv")


;output tree (dot file)
(provide toyout)
(define toyout "../output/toy-decision-tree.dot")

;reading input datasets
;read the csv file myfile as a list of strings
;with each line of the original file as an element of the list
;further split each line at commas
;so then we have a list of list of strings
(provide toy-raw)
(define toy-raw 
  (cdr (read-csv-file toytrain)))

(provide titanic-raw)
         (define titanic-raw (cdr (map (lambda(x) (cddr x )) (read-csv-file titanictrain))))


(provide mushroom-raw)
         (define mushroom-raw (cdr (read-csv-file mushroomtrain)))

;function to convert data to internal numerical format
;(features . result)
(provide format)
(define (format data)
 ( cons (map (lambda (x) (string->number x)) (cdr data)) (string->number (car data))))

;list of (features . result)
(provide toy)
(define toy
  (map (lambda (x) (format x)) toy-raw))

(provide titanic)
  (define titanic
  (map (lambda (x) (format x)) titanic-raw))



(provide mushroom)
(define mushroom
  (map (lambda (x) (format x)) mushroom-raw))


;============================================================================================================
;============================================================================================================
;============================================================================================================

;get fraction of result fields that are 1
;used to find probability value at leaf
(provide get-leaf-prob)
(define (get-leaf-prob data)
  (/ (apply + (map (lambda (x) (cdr x)) data)) (length data)))
 

;get entropy of dataset
(provide get-entropy)
(define (get-entropy data)
  (let* ([p (apply + (map (lambda (x) (cdr x)) data))]
         [n (- (length data) p)]
         [a1  (/ n (length data))]
         [a2  (/ p (length data))])
    (if(or (= 0 p) (= 0 n)) 0
       (- (* -1 a1 (log a1 2)) (* a2 (log a2 2))))))

;find the difference in entropy achieved
;by applying a decision function f to the data
(define (func f data)
  (sort (map (lambda (x) (cons (f (car x)) x)) data) (lambda (x y)(< (car x) (car y)))))
(define (rle l)
  (map (lambda(x) (map (lambda(y)(cdr y)) x))  (foldr (lambda (x y) (if(null? y) (list (list x))
                                                                       (if (eq? (car (caar y)) (car x)) (cons (cons x (car y)) (cdr y))
                                                                         (cons (list x) y)))) '() l)))
(define (split f data)
  (rle (func f data)))
(provide entropy-diff)
(define (entropy-diff f data)
  (let* ([l (split f data)]
         [len (length data)])
    (- (get-entropy data) (apply + (map (lambda(x) (* (/ (length x) len) (get-entropy x))) l)))))
  
(define (give l p)
  (if(null? l) (cdr p)
     (if(> (caar l) (car p)) (give (cdr l) (car l))
        (give (cdr l) p))))
;choose the decision function that most reduces entropy of the data
(provide choose-f)
(define (choose-f candidates data) ; returns a decision function
  (give (map (lambda (x) (cons (entropy-diff (cdr x) data) x)) candidates) (cons -5 0)))
  

(provide DTree)
(struct DTree (desc func kids)#:transparent)

;build a decision tree (depth limited) from the candidate decision functions and data
(provide build-tree)
(define (build-tree candidates data depth)
 
;(displayln (list data candidates))
  ;(if  (null? candidates)) '()
  ( if(or ( = 0 depth) (null? candidates)) (DTree (number->string (get-leaf-prob data)) '() '())
      (let* ([f (choose-f candidates data)]
             [rem (remove f candidates)]
             [pos (remove-duplicates (sort (map (lambda(x) ((cdr f) (car x))) data) <))])
        ;(displayln f)
        (DTree (cons (car f) pos) f (filter (lambda (x) (not (equal? x '()))) (map (lambda (x) (build-tree rem x (- depth 1))) (split (cdr f) data)))))))
  
(define (getid a l)
  (getid2 a l 0))
(define (getid2 a l id)
  (if(null? l) #f
  (if (equal? a (car l)) id
      (getid2 a (cdr l) (+ id 1)))))
;given a test data (features only), make a decision according to a decision tree
;returns probability of the test data being classified as 1
(provide make-decision)
(define (make-decision tree test)
  (match tree
    [(DTree a b c) (if (equal? c '()) (string->number a)
                       (let* ([val ((cdr b) test)]
                              [id (getid val (cdr a))])
                         
                         (if (not id) 0 (make-decision (list-ref c id) test))))]))
  

;============================================================================================================
;============================================================================================================
;============================================================================================================

;annotate list with indices
(define (pair-idx lst n)
  (if (empty? lst) `() (cons (cons (car lst) n) (pair-idx (cdr lst) (+ n 1))))
  )

;generate tree edges (parent to child) and recurse to generate sub trees
(define (dot-child children prefix tabs)
  (apply string-append
         (map (lambda (t)
                (string-append tabs
                               "r" prefix
                               "--"
                               "r" prefix "t" (~a (cdr t))
                               "[label=\"" (~a (cdr t)) "\"];" "\n"
                               (dot-helper (car t)
                                           (string-append prefix "t" (~a (cdr t)))
                                           (string-append tabs "\t")
                                           )
                               )
                ) children
                  )
         )
  )

;generate tree nodes and call function to generate edges
(define (dot-helper tree prefix tabs)
  (let* ([node (match tree [(DTree d f c) (cons d c)])]
         [d (car node)]
         [c (cdr node)])
    (string-append tabs
                   "r"
                   prefix
                   "[label=\"" d "\"];" "\n\n"
                   (dot-child (pair-idx c 0) prefix tabs)
                   )
    )
  )

;output tree (dot file)
(provide display-tree)
(define (display-tree tree outfile)
  (write-file outfile (string-append "graph \"decision-tree\" {" "\n"
                                     (dot-helper tree "" "\t")
                                     "}"
                                     )
              )
  )
;============================================================================================================
;============================================================================================================
;============================================================================================================

