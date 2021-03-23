function [int, a, b, c2, idx] = adapquad(f,a0,b0,tol0,maxit)
  int=0; 
  n=1; 
  a(1)=a0; 
  b(1)=b0; 
  tol(1)=tol0; 
  app(1)=trap(f,a,b);
  cnt = 0;
  c2 = 0;
  while n>0 % n is current position at end of the list
    cnt = cnt + 1;
    c=(a(n)+b(n))/2;
    oldapp=app(n);
    app(n)=trap(f,a(n),c);
    app(n+1)=trap(f,c,b(n));
    if abs(oldapp-(app(n)+app(n+1)))<3*tol(n)
      int=int+app(n)+app(n+1); % success
      n=n-1; % done with interval
      c2=c2+1;
      idx(c2) = n;
    else % divide into two intervals
      b(n+1)=b(n); 
      b(n)=c; % set up new intervals
      a(n+1)=c;
      tol(n)=tol(n)/2; 
      tol(n+1)=tol(n);
      n=n+1;
    end
    if cnt > maxit
      break
    end
  end
end

function s=trap(f,a,b)
  s=(f(a)+f(b))*(b-a)/2;
end