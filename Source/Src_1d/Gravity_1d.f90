! :::
! ::: ----------------------------------------------------------------
! :::

      subroutine ca_edge_interp(flo, fhi, nc, ratio, dir, &
           fine, fine_l0, fine_h0)
      implicit none
      integer flo(0:2-1), fhi(0:2-1), nc, ratio(0:2-1), dir
      integer fine_l0, fine_h0
      double precision fine(fine_l0:fine_h0,nc)
      integer i,n,M
      double precision val, df

!     Do linear in dir, pc transverse to dir, leave alone the fine values
!     lining up with coarse edges--assume these have been set to hold the 
!     values you want to interpolate to the rest.

      do n=1,nc
         do i=flo(0),fhi(0)-ratio(dir),ratio(0)
            df = fine(i+ratio(dir),n)-fine(i,n)
            do M=1,ratio(dir)-1
               val = fine(i,n) + df*dble(M)/dble(ratio(dir))
               fine(i+M,n) = val
            enddo                     
         enddo
      enddo

      end subroutine ca_edge_interp

! :::
! ::: ----------------------------------------------------------------
! :::

      subroutine ca_pc_edge_interp(lo, hi, nc, ratio, dir, &
           crse, crse_l0, crse_h0,  &
           fine, fine_l0, fine_h0)
      implicit none
      integer lo(1),hi(1), nc, ratio(0:2-1), dir
      integer crse_l0, crse_h0
      integer fine_l0, fine_h0
      double precision crse(crse_l0:crse_h0,nc)
      double precision fine(fine_l0:fine_h0,nc)
      integer i,ii,n

!     For edge-based data, fill fine values with piecewise-constant interp of coarse data.
!     Operate only on faces that overlap--ie, only fill the fine faces that make up each
!     coarse face, leave the in-between faces alone.
      do n=1,nc
         do i=lo(1),hi(1)
            ii = ratio(0)*i
            fine(ii,n) = crse(i,n)
         enddo
      enddo

      end subroutine ca_pc_edge_interp

! :::
! ::: ----------------------------------------------------------------
! :::

      subroutine ca_test_residual(lo, hi, &
           rhs, rhl1, rhh1,  &
           ecx, ecxl1, ecxh1, &
           dx, problo, coord_type)

      use bl_constants_module

      implicit none

      integer          :: lo(1),hi(1)
      integer          :: rhl1, rhh1
      integer          :: ecxl1, ecxh1
      integer          :: coord_type
      double precision :: rhs(rhl1:rhh1)
      double precision :: ecx(ecxl1:ecxh1)
      double precision :: dx(1),problo(1)

      double precision :: lapphi
      double precision :: rlo,rhi,rcen
      integer          :: i

      ! Cartesian 
      if (coord_type .eq. 0) then

         do i=lo(1),hi(1)
            lapphi = (ecx(i+1)-ecx(i)) / dx(1)
            rhs(i) = rhs(i) - lapphi
         enddo

      ! r-z
      else if (coord_type .eq. 1) then

         do i=lo(1),hi(1)
            rlo  = problo(1) + dble(i)*dx(1)
            rhi  = rlo + dx(1)
            rcen = HALF * (rlo+rhi)
            lapphi = (rhi*ecx(i+1)-rlo*ecx(i)) / (rcen*dx(1))
            rhs(i) = rhs(i) - lapphi
         enddo

      ! spherical
      else if (coord_type .eq. 2) then

         do i=lo(1),hi(1)
            rlo  = problo(1) + dble(i)*dx(1)
            rhi  = rlo + dx(1)
            rcen = HALF * (rlo+rhi)
            lapphi = (rhi**2 * ecx(i+1)-rlo**2 * ecx(i)) / (rcen**2 * dx(1))
            rhs(i) = rhs(i) - lapphi
         enddo

      else 
         print *,'Bogus coord_type in test_residual ' ,coord_type
         call bl_error("Error:: Gravity_1d.f90 :: ca_test_residual")
      end if

      end subroutine ca_test_residual

! :::
! ::: ----------------------------------------------------------------
! :::

      subroutine ca_compute_1d_grav(rho, r_l1, r_h1, grav, dx, problo)

      use fundamental_constants_module, only : Gconst
      use bl_constants_module

      implicit none

      integer         , intent(in   ) :: r_l1, r_h1
      double precision, intent(in   ) ::  rho(r_l1:r_h1)
      double precision, intent(  out) :: grav(r_l1:r_h1)
      double precision, intent(in   ) :: dx, problo(1)

      double precision, parameter ::  fourthirdspi = FOUR3RD * M_PI
      double precision :: rc,rlo,mass_encl,halfdx
      integer          :: i

      halfdx = HALF * dx

      do i = 0,r_h1
         rlo = problo(1) + dble(i) * dx
         rc  = rlo + halfdx
         if (i.gt.0) then
            mass_encl = mass_encl + fourthirdspi * halfdx * (rlo**2 + rlo*(rlo-halfdx) + (rlo-halfdx)**2) * rho(i-1) + &
                                    fourthirdspi * halfdx * ( rc**2 +  rc* rlo         +  rlo**2        ) * rho(i  )
         else
            mass_encl = fourthirdspi * halfdx * (rc**2 + rc*rlo +  rlo**2) * rho(i)
         end if
         grav(i) = -Gconst * mass_encl / rc**2
      enddo

      if (problo(1) .eq. ZERO) then
         do i = r_l1,-1
             grav(i) = -grav(-i-1)
         end do
      end if

      end subroutine ca_compute_1d_grav
