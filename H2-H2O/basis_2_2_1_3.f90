module basis
  implicit none
contains
  function emsav(x, c) result(v) bind(c, name="c_emsav")
    use iso_c_binding
    implicit none

    real (c_double), dimension(1:10) :: x
    real (c_double), dimension(0:101) :: c
    real (c_double)                   :: v
    ! ::::::::::::::::::::
    real (c_double), dimension(0:101) :: p

    call bemsav(x, p)
    v = dot_product(p,c)

    return
  end function emsav

  subroutine bemsav(x, p) bind(c, name="c_bemsav")
    use iso_c_binding
    implicit none

    real (c_double), dimension(1:10), intent(in)  :: x
    real (c_double), dimension(0:101), intent(out) :: p
    ! ::::::::::::::::::::
    real (c_double), dimension(0:38) :: m

    call evmono(x, m)
    call evpoly(m, p)

    return
  end subroutine bemsav

  subroutine evmono(x, m) bind(c, name="c_evmono")
    use iso_c_binding
    implicit none
    real (c_double), dimension(1:10), intent(in)  :: x
    real (c_double), dimension(0:38), intent(out) :: m
    !::::::::::::::::::::

    m(0) = 1.0D0
    m(1) = x(10)
    m(2) = x(9)
    m(3) = x(8)
    m(4) = x(7)
    m(5) = x(4)
    m(6) = x(6)
    m(7) = x(5)
    m(8) = x(3)
    m(9) = x(2)
    m(10) = x(1)
    m(11) = m(1)*m(2)
    m(12) = m(4)*m(5)
    m(13) = m(2)*m(6)
    m(14) = m(1)*m(7)
    m(15) = m(2)*m(8)
    m(16) = m(1)*m(9)
    m(17) = m(5)*m(6)
    m(18) = m(5)*m(7)
    m(19) = m(4)*m(8)
    m(20) = m(4)*m(9)
    m(21) = m(7)*m(8)
    m(22) = m(6)*m(9)
    m(23) = m(6)*m(8)
    m(24) = m(7)*m(9)
    m(25) = m(6)*m(7)
    m(26) = m(8)*m(9)
    m(27) = m(2)*m(17)
    m(28) = m(1)*m(18)
    m(29) = m(2)*m(19)
    m(30) = m(1)*m(20)
    m(31) = m(2)*m(23)
    m(32) = m(1)*m(24)
    m(33) = m(5)*m(25)
    m(34) = m(4)*m(26)
    m(35) = m(6)*m(21)
    m(36) = m(6)*m(24)
    m(37) = m(6)*m(26)
    m(38) = m(7)*m(26)

    return
  end subroutine evmono

  subroutine evpoly(m, p) bind(c, name="c_evpoly")
    use iso_c_binding
    implicit none
    real (c_double), dimension(0:38), intent(in)  :: m
    real (c_double), dimension(0:101), intent(out) :: p
    !::::::::::::::::::::

    p(0) = m(0)
    p(1) = m(1) + m(2)
    p(2) = m(3)
    p(3) = m(4) + m(5)
    p(4) = m(6) + m(7) + m(8) + m(9)
    p(5) = m(10)
    p(6) = m(11)
    p(7) = p(2)*p(1)
    p(8) = p(1)*p(3)
    p(9) = p(2)*p(3)
    p(10) = m(12)
    p(11) = m(13) + m(14) + m(15) + m(16)
    p(12) = p(1)*p(4) - p(11)
    p(13) = p(2)*p(4)
    p(14) = m(17) + m(18) + m(19) + m(20)
    p(15) = m(21) + m(22)
    p(16) = m(23) + m(24)
    p(17) = p(3)*p(4) - p(14)
    p(18) = m(25) + m(26)
    p(19) = p(5)*p(1)
    p(20) = p(2)*p(5)
    p(21) = p(5)*p(3)
    p(22) = p(5)*p(4)
    p(23) = p(1)*p(1) - p(6) - p(6)
    p(24) = p(2)*p(2)
    p(25) = p(3)*p(3) - p(10) - p(10)
    p(26) = p(4)*p(4) - p(18) - p(16) - p(15) - p(18) - p(16) - p(15)
    p(27) = p(5)*p(5)
    p(28) = p(2)*p(6)
    p(29) = p(6)*p(3)
    p(30) = p(2)*p(8)
    p(31) = p(10)*p(1)
    p(32) = p(2)*p(10)
    p(33) = p(6)*p(4)
    p(34) = p(2)*p(11)
    p(35) = p(2)*p(12)
    p(36) = m(27) + m(28) + m(29) + m(30)
    p(37) = p(1)*p(14) - p(36)
    p(38) = p(2)*p(14)
    p(39) = p(1)*p(15)
    p(40) = p(2)*p(15)
    p(41) = m(31) + m(32)
    p(42) = p(1)*p(16) - p(41)
    p(43) = p(2)*p(16)
    p(44) = p(3)*p(11) - p(36)
    p(45) = p(1)*p(17) - p(44)
    p(46) = p(2)*p(17)
    p(47) = p(10)*p(4)
    p(48) = p(3)*p(15)
    p(49) = p(3)*p(16)
    p(50) = p(1)*p(18)
    p(51) = p(2)*p(18)
    p(52) = m(33) + m(34)
    p(53) = m(35) + m(36) + m(37) + m(38)
    p(54) = p(3)*p(18) - p(52)
    p(55) = p(5)*p(6)
    p(56) = p(2)*p(19)
    p(57) = p(5)*p(8)
    p(58) = p(2)*p(21)
    p(59) = p(5)*p(10)
    p(60) = p(5)*p(11)
    p(61) = p(5)*p(12)
    p(62) = p(2)*p(22)
    p(63) = p(5)*p(14)
    p(64) = p(5)*p(15)
    p(65) = p(5)*p(16)
    p(66) = p(5)*p(17)
    p(67) = p(5)*p(18)
    p(68) = p(6)*p(1)
    p(69) = p(2)*p(23)
    p(70) = p(2)*p(7)
    p(71) = p(3)*p(23)
    p(72) = p(2)*p(9)
    p(73) = p(1)*p(25)
    p(74) = p(2)*p(25)
    p(75) = p(10)*p(3)
    p(76) = p(1)*p(11) - p(33)
    p(77) = p(1)*p(12) - p(33)
    p(78) = p(2)*p(13)
    p(79) = p(3)*p(14) - p(47)
    p(80) = p(3)*p(17) - p(47)
    p(81) = p(4)*p(11) - p(50) - p(41) - p(39) - p(41)
    p(82) = p(1)*p(26) - p(81)
    p(83) = p(2)*p(26)
    p(84) = p(4)*p(14) - p(52) - p(49) - p(48) - p(52)
    p(85) = p(4)*p(15) - p(53)
    p(86) = p(4)*p(16) - p(53)
    p(87) = p(3)*p(26) - p(84)
    p(88) = p(4)*p(18) - p(53)
    p(89) = p(5)*p(23)
    p(90) = p(2)*p(20)
    p(91) = p(5)*p(25)
    p(92) = p(5)*p(26)
    p(93) = p(5)*p(19)
    p(94) = p(2)*p(27)
    p(95) = p(5)*p(21)
    p(96) = p(5)*p(22)
    p(97) = p(1)*p(23) - p(68)
    p(98) = p(2)*p(24)
    p(99) = p(3)*p(25) - p(75)
    p(100) = p(4)*p(26) - p(88) - p(86) - p(85)
    p(101) = p(5)*p(27)

    return
  end subroutine evpoly

end module basis
