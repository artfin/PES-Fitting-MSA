module basis
  implicit none
contains
  function emsav(x, c) result(v) bind(c, name="c_emsav")
    use iso_c_binding
    implicit none

    real (c_double), dimension(1:21) :: x
    real (c_double), dimension(0:31) :: c
    real (c_double)                   :: v
    ! ::::::::::::::::::::
    real (c_double), dimension(0:31) :: p

    call bemsav(x, p)
    v = dot_product(p,c)

    return
  end function emsav

  subroutine bemsav(x, p) bind(c, name="c_bemsav")
    use iso_c_binding
    implicit none

    real (c_double), dimension(1:21), intent(in)  :: x
    real (c_double), dimension(0:31), intent(out) :: p
    ! ::::::::::::::::::::
    real (c_double), dimension(0:139) :: m

    call evmono(x, m)
    call evpoly(m, p)

    return
  end subroutine bemsav

  subroutine evmono(x, m) bind(c, name="c_evmono")
    use iso_c_binding
    implicit none
    real (c_double), dimension(1:21), intent(in)  :: x
    real (c_double), dimension(0:139), intent(out) :: m
    !::::::::::::::::::::

    m(0) = 1.0D0
    m(1) = x(21)
    m(2) = x(20)
    m(3) = x(19)
    m(4) = x(18)
    m(5) = x(15)
    m(6) = x(11)
    m(7) = x(6)
    m(8) = x(17)
    m(9) = x(16)
    m(10) = x(14)
    m(11) = x(13)
    m(12) = x(10)
    m(13) = x(9)
    m(14) = x(5)
    m(15) = x(4)
    m(16) = x(12)
    m(17) = x(8)
    m(18) = x(7)
    m(19) = x(3)
    m(20) = x(2)
    m(21) = x(1)
    m(22) = m(1)*m(2)
    m(23) = m(4)*m(5)
    m(24) = m(4)*m(6)
    m(25) = m(5)*m(6)
    m(26) = m(4)*m(7)
    m(27) = m(5)*m(7)
    m(28) = m(6)*m(7)
    m(29) = m(2)*m(8)
    m(30) = m(1)*m(9)
    m(31) = m(2)*m(10)
    m(32) = m(1)*m(11)
    m(33) = m(2)*m(12)
    m(34) = m(1)*m(13)
    m(35) = m(2)*m(14)
    m(36) = m(1)*m(15)
    m(37) = m(5)*m(8)
    m(38) = m(5)*m(9)
    m(39) = m(4)*m(10)
    m(40) = m(4)*m(11)
    m(41) = m(6)*m(8)
    m(42) = m(6)*m(9)
    m(43) = m(6)*m(10)
    m(44) = m(6)*m(11)
    m(45) = m(4)*m(12)
    m(46) = m(5)*m(12)
    m(47) = m(4)*m(13)
    m(48) = m(5)*m(13)
    m(49) = m(7)*m(8)
    m(50) = m(7)*m(9)
    m(51) = m(7)*m(10)
    m(52) = m(7)*m(11)
    m(53) = m(7)*m(12)
    m(54) = m(7)*m(13)
    m(55) = m(4)*m(14)
    m(56) = m(5)*m(14)
    m(57) = m(6)*m(14)
    m(58) = m(4)*m(15)
    m(59) = m(5)*m(15)
    m(60) = m(6)*m(15)
    m(61) = m(9)*m(10)
    m(62) = m(8)*m(11)
    m(63) = m(9)*m(12)
    m(64) = m(11)*m(12)
    m(65) = m(8)*m(13)
    m(66) = m(10)*m(13)
    m(67) = m(9)*m(14)
    m(68) = m(11)*m(14)
    m(69) = m(13)*m(14)
    m(70) = m(8)*m(15)
    m(71) = m(10)*m(15)
    m(72) = m(12)*m(15)
    m(73) = m(8)*m(10)
    m(74) = m(9)*m(11)
    m(75) = m(8)*m(12)
    m(76) = m(10)*m(12)
    m(77) = m(9)*m(13)
    m(78) = m(11)*m(13)
    m(79) = m(8)*m(14)
    m(80) = m(10)*m(14)
    m(81) = m(12)*m(14)
    m(82) = m(9)*m(15)
    m(83) = m(11)*m(15)
    m(84) = m(13)*m(15)
    m(85) = m(8)*m(9)
    m(86) = m(10)*m(11)
    m(87) = m(12)*m(13)
    m(88) = m(14)*m(15)
    m(89) = m(6)*m(16)
    m(90) = m(5)*m(17)
    m(91) = m(4)*m(18)
    m(92) = m(7)*m(16)
    m(93) = m(7)*m(17)
    m(94) = m(7)*m(18)
    m(95) = m(5)*m(19)
    m(96) = m(6)*m(19)
    m(97) = m(4)*m(20)
    m(98) = m(6)*m(20)
    m(99) = m(4)*m(21)
    m(100) = m(5)*m(21)
    m(101) = m(12)*m(16)
    m(102) = m(13)*m(16)
    m(103) = m(10)*m(17)
    m(104) = m(11)*m(17)
    m(105) = m(8)*m(18)
    m(106) = m(9)*m(18)
    m(107) = m(14)*m(16)
    m(108) = m(14)*m(17)
    m(109) = m(14)*m(18)
    m(110) = m(15)*m(16)
    m(111) = m(15)*m(17)
    m(112) = m(15)*m(18)
    m(113) = m(10)*m(19)
    m(114) = m(11)*m(19)
    m(115) = m(12)*m(19)
    m(116) = m(13)*m(19)
    m(117) = m(8)*m(20)
    m(118) = m(9)*m(20)
    m(119) = m(12)*m(20)
    m(120) = m(13)*m(20)
    m(121) = m(8)*m(21)
    m(122) = m(9)*m(21)
    m(123) = m(10)*m(21)
    m(124) = m(11)*m(21)
    m(125) = m(18)*m(19)
    m(126) = m(17)*m(20)
    m(127) = m(16)*m(21)
    m(128) = m(16)*m(17)
    m(129) = m(16)*m(18)
    m(130) = m(17)*m(18)
    m(131) = m(16)*m(19)
    m(132) = m(17)*m(19)
    m(133) = m(16)*m(20)
    m(134) = m(18)*m(20)
    m(135) = m(19)*m(20)
    m(136) = m(17)*m(21)
    m(137) = m(18)*m(21)
    m(138) = m(19)*m(21)
    m(139) = m(20)*m(21)

    return
  end subroutine evmono

  subroutine evpoly(m, p) bind(c, name="c_evpoly")
    use iso_c_binding
    implicit none
    real (c_double), dimension(0:139), intent(in)  :: m
    real (c_double), dimension(0:31), intent(out) :: p
    !::::::::::::::::::::

    p(0) = m(0)
    p(1) = m(1) + m(2)
    p(2) = m(3)
    p(3) = m(4) + m(5) + m(6) + m(7)
    p(4) = m(8) + m(9) + m(10) + m(11) + m(12) + m(13) &
	&  + m(14) + m(15)
    p(5) = m(16) + m(17) + m(18) + m(19) + m(20) + m(21)
    p(6) = m(22)
    p(7) = p(2)*p(1)
    p(8) = p(1)*p(3)
    p(9) = p(2)*p(3)
    p(10) = m(23) + m(24) + m(25) + m(26) + m(27) + m(28)
    p(11) = m(29) + m(30) + m(31) + m(32) + m(33) + m(34) &
	&  + m(35) + m(36)
    p(12) = p(1)*p(4) - p(11)
    p(13) = p(2)*p(4)
    p(14) = m(37) + m(38) + m(39) + m(40) + m(41) + m(42) &
	&  + m(43) + m(44) + m(45) + m(46) + m(47) &
	&  + m(48) + m(49) + m(50) + m(51) + m(52) &
	&  + m(53) + m(54) + m(55) + m(56) + m(57) &
	&  + m(58) + m(59) + m(60)
    p(15) = m(61) + m(62) + m(63) + m(64) + m(65) + m(66) &
	&  + m(67) + m(68) + m(69) + m(70) + m(71) &
	&  + m(72)
    p(16) = m(73) + m(74) + m(75) + m(76) + m(77) + m(78) &
	&  + m(79) + m(80) + m(81) + m(82) + m(83) &
	&  + m(84)
    p(17) = p(3)*p(4) - p(14)
    p(18) = m(85) + m(86) + m(87) + m(88)
    p(19) = p(1)*p(5)
    p(20) = p(2)*p(5)
    p(21) = m(89) + m(90) + m(91) + m(92) + m(93) + m(94) &
	&  + m(95) + m(96) + m(97) + m(98) + m(99) &
	&  + m(100)
    p(22) = m(101) + m(102) + m(103) + m(104) + m(105) + m(106) &
	&  + m(107) + m(108) + m(109) + m(110) + m(111) &
	&  + m(112) + m(113) + m(114) + m(115) + m(116) &
	&  + m(117) + m(118) + m(119) + m(120) + m(121) &
	&  + m(122) + m(123) + m(124)
    p(23) = m(125) + m(126) + m(127)
    p(24) = p(3)*p(5) - p(21)
    p(25) = p(4)*p(5) - p(22)
    p(26) = m(128) + m(129) + m(130) + m(131) + m(132) + m(133) &
	&  + m(134) + m(135) + m(136) + m(137) + m(138) &
	&  + m(139)
    p(27) = p(1)*p(1) - p(6) - p(6)
    p(28) = p(2)*p(2)
    p(29) = p(3)*p(3) - p(10) - p(10)
    p(30) = p(4)*p(4) - p(18) - p(16) - p(15) - p(18) - p(16) - p(15)
    p(31) = p(5)*p(5) - p(26) - p(23) - p(26) - p(23)

    return
  end subroutine evpoly

end module basis
