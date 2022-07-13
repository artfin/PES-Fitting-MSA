#!/usr/bin/perl -w
use strict;
use List::Util qw/sum/;

die " Syntax:\n   ./derivative [wdir] [max degree] [symmetry]\n" unless $#ARGV > 0;

my $wdir   = shift(@ARGV);
my $degree = shift(@ARGV);
my @atom   = @ARGV;

my $nat = sum @atom;
my $ng = 3*$nat;

my $stub = "$wdir/MOL_".join("_",@atom)."_$degree";
my $mono = $stub.".MONO";
my $poly = $stub.".POLY";
my $fbas = "$wdir/f_gradbasis_".join("_", @atom)."_$degree.f90";

my $c_bind = 1; # = false

# ... read in the MONO data file
my @datMono = ();
open(MONO,$mono)  or die "Can not open $mono to read!\n";
while(<MONO>){
    chomp;
    my @mono = split;
    push(@datMono,\@mono);
}
close(MONO);

my $nx = $#{$datMono[0]};
my $nm = $#datMono;

# ... read in the POLY data file
my @datPoly = ();
open(POLY,$poly)  or die "Can not open $poly to read!\n";
while(<POLY>){
    chomp;
    my @poly = split;
    push(@datPoly,\@poly);
}
close(POLY);

my $np = $#datPoly;

# ... record the basis function bemsa

open(OUT,">$fbas") or die "Can not open $fbas to write!\n";
print_module_head(\*OUT,$nx,$nm,$np);

print_mono_head(\*OUT,$nx,$nm);
for(my $i=0;$i<=$nm;$i++){print_mono_body(\*OUT, $i, $datMono[$i]);}
print_mono_foot(\*OUT);

print_poly_head(\*OUT, $nm, $np);
for(my $i=0;$i<=$np;$i++){print_poly_body(\*OUT, $i, $datPoly[$i]);}
print_poly_foot(\*OUT);
print_module_foot(\*OUT);
close(OUT);

sub print_module_head {
    my ($out,$nx,$nm,$np) = @_;
    print $out "module gradient\n";
    print $out "  implicit none\n";
    print $out "\n";
    print $out "contains\n";
}

sub print_module_foot {
    my ($out) = @_;
    print $out "end module gradient\n";
}

sub print_mono_head {
    my ($om, $nx, $nm)= @_;
    
    if ($c_bind) {
        print $om "  subroutine devmono(drdx, dm, m, flag) bind(c, name=\"c_devmono\")\n";
        print $om "    use iso_c_binding\n";
    } else {
        print $om "  subroutine devmono(drdx, dm, m, flag)\n";
    }

    print $om "    implicit none\n";
    
    if ($c_bind) {
        print $om "    real (c_double), dimension($ng,$nx), intent(in)  :: drdx\n";
        print $om "    real (c_double), dimension(0:$nm),   intent(out) :: dm\n";
        print $om "    real (c_double), dimension(0:$nm),   intent(in)  :: m\n";

        # explicitly stating that integer::flag will be passed by value
        # (instead of through pointer by default)
        print $om "    integer (c_int), intent(in), VALUE  :: flag\n"; 
    } else {
        print $om "    real, dimension($ng,$nx), intent(in)  :: drdx\n";
        print $om "    real, dimension(0:$nm),   intent(out) :: dm\n";
        print $om "    real, dimension(0:$nm),   intent(in)  :: m\n";
    }
    print $om "    !::::::::::::::::::::\n";
    print $om "    real::a\n";
    print $om "\n";

    # TODO: move this and diffentiation of Morse variables outside
    #       of the devmono procedure
    print $om "    a = 2.5d0\n"; 
    print $om "\n";

}

sub print_poly_head{
    my ($op, $nm, $np)= @_;

    if ($c_bind) {
        print $op "  subroutine devpoly(dm, p, dp) bind(c, name=\"c_devpoly\")\n";
        print $op "    use iso_c_binding\n";
    } else {
        print $op "  subroutine devpoly(dm, p, dp)\n";
    }
    
    print $op "    implicit none\n";

    if ($c_bind) {
        print $op "   real (c_double), dimension(0:$nm), intent(in)  :: dm\n";
        print $op "   real (c_double), dimension(0:$np), intent(in)  :: p\n";
        print $op "   real (c_double), dimension(0:$np), intent(out) :: dp\n"; 
    } else {
        print $op "    real, dimension(0:$nm), intent(in)  :: dm\n";
        print $op "    real, dimension(0:$np), intent(in)  :: p\n";
        print $op "    real, dimension(0:$np), intent(out) :: dp\n";
    }

    print $op "    !::::::::::::::::::::\n";
    print $op "\n";
    
}

sub print_mono_foot{
    my ($om)= @_;
    
    print $om "\n";
    print $om "    return\n";
    print $om "  end subroutine devmono\n";
    print $om "\n";
    
}

sub print_poly_foot{
    my ($op)= @_;
    
    print $op "\n";
    print $op "    return\n";
    print $op "  end subroutine devpoly\n";
    print $op "\n";
}

sub print_mono_body{
    my ($om, $iMono, $mono)= @_;
    
    printf $om "    dm(%d) = ", $iMono;
    
    my $stat = shift(@$mono);
    if($stat == 0){
	my $new = 1;
	my $nz = 0;
	for(my $i=0;$i<=$#$mono;$i++) {
	    my $deg = $mono->[$i];
	    next if $deg == 0;
	    $nz++;

        # TODO: move this differentiation of Morse varibles outside of the devmono proc
        #
	    if($deg == 1) {
		  $new? printf $om "-m(%d)/a*drdx(flag,%d)",$iMono,$i+1: printf $om "-m(%d)/a*drdx(flag,%d)",$iMono,$i+1;
		  $new = 0;
	    } else {
		  $new? printf $om "-$deg*m(%d)/a*drdx(flag,%d)",$iMono,$i+1 : printf $om "-$deg*m(%d)/a*drdx(flag,%d)",$iMono,$i+1;
		  $new = 0;
	    }
	    print $om " &\n\t& " if $nz % 5 == 0 && $i<$#$mono;
	}
        print $om "0.0D0" if $new;
    } else {
	    printf $om "dm(%d)",$mono->[0];
        printf $om "*m(%d)",$mono->[1];
        printf $om " + m(%d)",$mono->[0];
        printf $om "*dm(%d)",$mono->[1];
    }
    print $om "\n";
    unshift(@$mono,$stat);
}

sub print_poly_body{
    my ($op, $iPoly, $poly)= @_;
    
    my $stat = shift(@$poly);
    printf $op "    dp(%d) = ",$iPoly;
    if($stat == 2){		# sum of mono
	printf $op "dm(%d)",$poly->[0];
	for(my $i=1;$i<=$#$poly;$i++){
	    my $idx = $poly->[$i];
	    printf $op " + dm(%d)",$idx;
	    print $op " &\n\t& " if $i % 5 == 0 && $i<$#$poly;
	}
    }else{			# usable decomp
	my $ima = shift(@$poly);
	my $imb = shift(@$poly);
	printf $op "dp(%d)*p(%d) + p(%d)*dp(%d)", $ima, $imb, $ima, $imb;
	for(my $i=0;$i<=$#$poly;$i++){
	    my $idx = $poly->[$i];
	    printf $op " - dp(%d)",$idx;
	    print $op " &\n\t& " if $i % 5 == 0 && $i<$#$poly && $i > 0;
	}
	
	unshift(@$poly,$imb);
	unshift(@$poly,$ima);
    }
    print $op "\n";
    unshift(@$poly,$stat);
}
