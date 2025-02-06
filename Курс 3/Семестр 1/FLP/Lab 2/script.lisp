; 3) Заменяющую в списке все вхождения x на y.
; Например, x=1, y=+, L = (2 1 3 5 1 1 8) –> (2 + 3 5 + + 8).

(print "Task 1")

(defun task_1(lst x y)
  (cond ((null lst) nil)
        ((equal (car lst) x) (cons y(task_1(cdr lst)x y)))
        (t (cons (car lst)(task_1(cdr lst) x y)))
  )
)

(defparameter list1 '((nil) (nil) nil))
(print (task_1 list1 'nil '(nil)))



; 13) Определяющую, сколько раз заданное s-выражение входит в список.
; Например, x=(a), L=(1 (a) x (a) 2 a 1 2 d) –> 2.

(terpri)
(print "Task 2")

(defun task_2(lst x)
  (cond ((null lst) 0)
        ((equal (car lst) x) (+ 1 (task_2(cdr lst) x)))
        (t (task_2(cdr lst) x))
  )
)

(defparameter list2 '(nil))
(print (task_2 list2 'nil))



; 23) Формирующую подсписок из n элементов списка L, начиная с k-го элемента.
; (нумерация элементов должна начинаться с 1).
; Например, L=(-2 6 s -1 4 f 0 z x r), k=3, n=4 –> (s -1 4 f).

(terpri)
(print "Task 3")

(defun task_3(lst k n)
  (cond ((or (null lst)(<= n 0)) nil)
        ((> k 1) (task_3(cdr lst) (- k 1) n))
        (t (cons (car lst)(task_3(cdr lst) 1 (- n 1))))
  )
)

(defparameter list3 '(1 2))
(print (task_3 list3 1 -3))

